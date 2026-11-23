# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Vocabulary layout and eager rollout support for sharded OPSD logits."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

import torch


@dataclass(frozen=True)
class VocabShardLayout:
    vocab_size: int
    start: int
    end: int
    ranks: tuple[int, ...]
    ranges: tuple[tuple[int, int], ...]
    group: Any = field(compare=False, repr=False)

    @property
    def local_size(self) -> int:
        return self.end - self.start

    @classmethod
    def from_model(cls, model) -> "VocabShardLayout":
        from deepspeed import comm as dist

        error = None
        try:
            head = model.get_output_embeddings()
            if head is None:
                raise ValueError("The model must expose its output embedding")
            sharded = getattr(head, "is_vocab_parallel_lm_head", False)
            group = getattr(head, "mp_group", None) if sharded else None
            size = dist.get_world_size(group) if group is not None else 1
            weight_shape = getattr(head.weight, "ds_shape", head.weight.shape)
            vocab = int(head.vocab_size if sharded else weight_shape[0])
            start = int(head.vocab_start_index) if sharded else 0
            end = int(head.vocab_end_index) if sharded else vocab
            width = int(weight_shape[0])
        except (ValueError, AttributeError, IndexError, TypeError) as failure:
            error = str(failure)
        # Check local metadata before TP peers enter the tensor collective.
        # A malformed head on one rank must produce a uniform startup error.
        _raise_if_any_error(error, "Vocabulary head metadata")
        rank = dist.get_rank() if dist.is_initialized() else 0
        local = torch.tensor([rank, vocab, start, end, width],
                             device=head.weight.device, dtype=torch.int64)
        gathered = [torch.empty_like(local) for _ in range(size)]
        if size > 1:
            dist.all_gather(gathered, local, group=group)
        else:
            gathered[0].copy_(local)
        metadata = [entry.tolist() for entry in gathered]
        cursor, error = 0, None
        for _, total, left, right, width in metadata:
            if total != vocab or left != cursor or right <= left or right - left != width:
                error = "Vocabulary shards must form a nonempty, contiguous partition"
            cursor = right
        if cursor != vocab:
            error = "Vocabulary shards do not cover the model vocabulary"
        _raise_if_any_error(error, "Vocabulary partition")
        return cls(vocab, start, end, tuple(m[0] for m in metadata),
                   tuple((m[2], m[3]) for m in metadata), group)

    def require_compatible(self, other: "VocabShardLayout") -> None:
        if (self.vocab_size, self.ranks, self.ranges, self.start, self.end) != (
                other.vocab_size, other.ranks, other.ranges, other.start, other.end):
            raise ValueError("Teacher and student require the same ordered TP vocabulary partition")


def _agree_metadata(entry):
    from deepspeed import comm as dist

    if not dist.is_initialized():
        return [entry]
    entries = [None] * dist.get_world_size()
    dist.all_gather_object(entries, entry)
    return entries


def _raise_if_any_error(error, context):
    errors = [item["error"] for item in _agree_metadata({"error": error}) if item["error"]]
    if errors:
        raise ValueError(context + " failed: " + "; ".join(sorted(set(errors))))


def tokenizer_fingerprint(tokenizer):
    semantic = {"vocab": tokenizer.get_vocab(), "added": tokenizer.get_added_vocab(),
                "special_ids": {name: getattr(tokenizer, name, None)
                                for name in ["pad_token_id", "eos_token_id", "bos_token_id", "unk_token_id"]}}
    return hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()


def prepare_vocab_parallel(student, teacher, student_tokenizer, teacher_tokenizer):
    student_layout = VocabShardLayout.from_model(student)
    teacher_layout = VocabShardLayout.from_model(teacher)
    error = None
    try:
        student_layout.require_compatible(teacher_layout)
        if tokenizer_fingerprint(student_tokenizer) != tokenizer_fingerprint(teacher_tokenizer):
            raise ValueError("Teacher and student token IDs have different vocabulary semantics")
    except ValueError as failure:
        error = str(failure)
    errors = [item["error"] for item in _agree_metadata({"error": error}) if item["error"]]
    if errors:
        raise ValueError("Vocabulary-parallel preflight failed: " + "; ".join(sorted(set(errors))))
    return student_layout, teacher_layout


def validate_tp_batch(layout, input_ids, attention_mask, response_mask):
    from deepspeed import comm as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    digests = [hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
               for value in [input_ids, attention_mask, response_mask]]
    entry = {"rank": rank, "ranks": layout.ranks, "shape": list(input_ids.shape), "digests": digests}
    entries = _agree_metadata(entry)
    by_rank = {item["rank"]: item for item in entries}
    for item in entries:
        for peer in item["ranks"]:
            if (by_rank[peer]["shape"], by_rank[peer]["digests"]) != (item["shape"], item["digests"]):
                raise ValueError("Input tokens and masks must agree within each TP group")


def validate_training_logits(layout, student_logits, teacher_cache):
    error = None
    try:
        layout.require_compatible(teacher_cache.layout)
        if student_logits.ndim != 3 or tuple(student_logits.shape) != teacher_cache.shape:
            raise ValueError("Teacher cache and student logits require identical local shapes")
        if student_logits.shape[-1] != layout.local_size:
            raise ValueError("Model output differs from the validated local vocabulary width")
    except (ValueError, AttributeError) as failure:
        error = str(failure)
    errors = [item["error"] for item in _agree_metadata({"error": error}) if item["error"]]
    if errors:
        raise ValueError("Vocabulary-parallel forward failed: " + "; ".join(sorted(set(errors))))


@contextmanager
def gathered_rollout_logits(model, layout: VocabShardLayout):
    """Expose only the last token's full vocabulary during eager generation.

    Training and teacher forwards keep their local output shards. The hook
    neither replaces parameters nor changes the head's gather configuration.
    """
    from deepspeed import comm as dist

    head = model.get_output_embeddings()
    if getattr(head, "_opsd_rollout_gather_active", False):
        raise RuntimeError("Vocabulary rollout scopes cannot be nested")

    def gather_last_token(module, args, output):
        if model.training or torch.is_grad_enabled():
            raise RuntimeError("Vocabulary rollout gathering requires eval and no_grad")
        if output.ndim != 3 or output.shape[-1] != layout.local_size:
            raise ValueError("Rollout head output does not match its vocabulary layout")
        last = output[:, -1:, :].contiguous()
        if len(layout.ranks) == 1:
            return last
        widths = [right - left for left, right in layout.ranges]
        padded = torch.nn.functional.pad(last, (0, max(widths) - last.shape[-1]))
        shards = [torch.empty_like(padded) for _ in widths]
        dist.all_gather(shards, padded, group=layout.group)
        return torch.cat([shard[..., :width] for shard, width in zip(shards, widths)], dim=-1)

    head._opsd_rollout_gather_active = True
    handle = None
    try:
        handle = head.register_forward_hook(gather_last_token)
        yield
    finally:
        if handle is not None:
            handle.remove()
        del head._opsd_rollout_gather_active
