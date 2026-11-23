# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Real AutoTP model probe; run with torchrun on two GPUs.

Random tiny Qwen2 weights qualify integration mechanics, not pretrained
distillation quality or representative training performance.
"""

import argparse
import copy
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import deepspeed
from deepspeed import comm as dist
import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from deepspeed.runtime.rollout import RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout
from deepspeed.sequence.vocab_parallel_ops import vocab_parallel_kl_div
from vocab_parallel import VocabShardLayout, gathered_rollout_logits


class GatheredReference(torch.autograd.Function):
    """Dense oracle on the same TP layout; one replicated loss per TP rank."""

    @staticmethod
    def forward(ctx, local_logits, layout):
        ctx.start, ctx.end = layout.start, layout.end
        widths = [right - left for left, right in layout.ranges]
        padded = F.pad(local_logits.contiguous(), (0, max(widths) - local_logits.shape[-1]))
        shards = [torch.empty_like(padded) for _ in widths]
        dist.all_gather(shards, padded, group=layout.group)
        return torch.cat([value[..., :width] for value, width in zip(shards, widths)], -1)

    @staticmethod
    def backward(ctx, gradient):
        return gradient[..., ctx.start:ctx.end].contiguous(), None


def make_model(dtype, tied, width, layers, vocab_size=64):
    config = Qwen2Config(vocab_size=vocab_size, hidden_size=width, intermediate_size=width * 2,
                         num_hidden_layers=layers, num_attention_heads=4,
                         num_key_value_heads=4, max_position_embeddings=64,
                         attention_dropout=0.0, tie_word_embeddings=tied,
                         pad_token_id=0, eos_token_id=1, bos_token_id=2)
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config).to(dtype=dtype)


def engine_config(dtype, train):
    config = {"train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": 2 if train else 1,
              "tensor_parallel": {"autotp_size": 2, "vocab_parallel_lm_head": True},
              "zero_optimization": {"stage": 0}, "gradient_clipping": 0.0,
              "bf16": {"enabled": dtype == torch.bfloat16}}
    if train:
        config["optimizer"] = {"type": "SGD", "params": {"lr": 0.05}}
    return config


def compare(actual, expected, dtype, what):
    atol, rtol = (1e-5, 1e-4) if dtype == torch.float32 else (2e-3, 1e-2)
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol,
                               msg=lambda message: what + ": " + message)
    return float((actual.float() - expected.float()).abs().max().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--tied", action="store_true")
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False,
                              timeout=timedelta(seconds=120))
    rank = dist.get_rank()
    assert dist.get_world_size() == 2, "This probe requires TP2, DP1"
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(1781)
    student = make_model(dtype, args.tied, 32, 1)
    teacher = make_model(dtype, args.tied, 64, 2)
    prompts = torch.tensor([[2, 3, 4]], device="cuda")
    mask = torch.ones_like(prompts)

    # Untied heads force the initial global maximum onto the second shard.
    if not args.tied:
        student.eval()
        with torch.no_grad():
            hidden = student.model(prompts.cpu()).last_hidden_state[0, -1]
            student.lm_head.weight.zero_()
            student.lm_head.weight[48].copy_(hidden)
    reference = copy.deepcopy(student).cuda()
    teacher_reference = copy.deepcopy(teacher).cuda().eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in teacher_reference.parameters():
        parameter.requires_grad_(False)
    student_engine, *_ = deepspeed.initialize(model=student, model_parameters=student.parameters(),
                                             config=engine_config(dtype, True))
    teacher_engine, *_ = deepspeed.initialize(model=teacher, config=engine_config(dtype, False))
    student_layout = VocabShardLayout.from_model(student_engine.module)
    teacher_layout = VocabShardLayout.from_model(teacher_engine.module)
    student_layout.require_compatible(teacher_layout)
    reference_engine = None
    if dtype == torch.bfloat16:
        # BF16 TP GEMMs and reductions round differently from a full-width
        # unsharded forward. Pair the real engines on the same layout and
        # optimizer, while the reference loss still uses ordinary dense math.
        reference_engine, *_ = deepspeed.initialize(model=reference, model_parameters=reference.parameters(),
                                                    config=engine_config(dtype, True))
        reference = reference_engine.module
        reference_layout = VocabShardLayout.from_model(reference)
        student_layout.require_compatible(reference_layout)
        reference.get_output_embeddings().register_forward_hook(
            lambda module, args, output: GatheredReference.apply(output, reference_layout))
    head = student_engine.module.get_output_embeddings()
    parameter_id, storage_ptr = id(head.weight), head.weight.data_ptr()
    if args.tied:
        assert student_engine.module.get_input_embeddings().weight is head.weight
    rollout = HybridEngineRollout(student_engine, SimpleNamespace(pad_token_id=0, eos_token_id=1))
    records = []

    def generate_and_compare(tag):
        student_engine.eval()
        reference.eval()
        with torch.no_grad():
            expected = reference.generate(prompts, attention_mask=mask, max_new_tokens=3,
                                          eos_token_id=None, pad_token_id=0, do_sample=False)
            with gathered_rollout_logits(student_engine.module, student_layout):
                actual = rollout.generate(RolloutRequest(prompts, mask),
                                          SamplingConfig(max_new_tokens=3, temperature=0.0)).input_ids
            # Match the rollout's EOS trimming, rather than hiding EOS differences.
            expected, _ = rollout._pad_after_eos(expected, response_start=prompts.shape[1],
                                                eos_token_id=1, pad_token_id=0)
            assert torch.equal(actual, expected), (tag, actual.tolist(), expected.tolist())
            if tag == "before" and not args.tied:
                assert int(actual[0, prompts.shape[1]]) == 48
            assert id(head.weight) == parameter_id and head.weight.data_ptr() == storage_ptr
            assert not getattr(head, "_opsd_rollout_gather_active", False)
            sharded = student_engine.module(input_ids=actual, attention_mask=torch.ones_like(actual)).logits
            assert sharded.shape[-1] == student_layout.local_size
        records.append({"case": tag + "_global_rollout", "tokens": actual.tolist(), "status": "PASS"})
        return actual

    input_ids = generate_and_compare("before")
    attention = torch.ones_like(input_ids)
    response_mask = torch.zeros_like(input_ids[:, 1:], dtype=torch.float32)
    response_mask[:, prompts.shape[1] - 1:] = 1
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    reference.train()
    teacher_engine.eval()
    for update in range(5):
        for microstep in range(2):
            student_engine.train()
            output = student_engine(input_ids=input_ids, attention_mask=attention).logits[:, :-1].float()
            with torch.no_grad():
                target = teacher_engine(input_ids=input_ids, attention_mask=attention).logits[:, :-1].float()
                target_full = (GatheredReference.apply(target, teacher_layout) if reference_engine is not None else
                               teacher_reference(input_ids=input_ids, attention_mask=attention).logits[:, :-1].float())
            ref_logits = reference(input_ids=input_ids, attention_mask=attention).logits[:, :-1].float()
            compare(output, ref_logits[..., student_layout.start:student_layout.end], dtype, "student shard")
            compare(target, target_full[..., teacher_layout.start:teacher_layout.end], dtype, "teacher shard")
            for reverse in (False, True):
                candidate_tokens = vocab_parallel_kl_div(output, target, reverse=reverse,
                                                        tp_group=student_layout.group)
                p, q = F.log_softmax(ref_logits, -1), F.log_softmax(target_full, -1)
                expected_tokens = (p.exp() * (p - q)).sum(-1) if reverse else (q.exp() * (q - p)).sum(-1)
                error = compare(candidate_tokens, expected_tokens, dtype, "KL direction")
                records.append({"case": "KL", "update": update, "microstep": microstep,
                                "reverse": reverse, "max_error": error, "status": "PASS"})
            loss = (candidate_tokens * response_mask).sum() / response_mask.sum()
            ref_loss = (expected_tokens * response_mask).sum() / response_mask.sum()
            student_engine.backward(loss)
            if reference_engine is None:
                (ref_loss / 2).backward()
            else:
                reference_engine.backward(ref_loss)
            student_engine.step()
            if reference_engine is not None:
                reference_engine.step()
        if reference_engine is None:
            optimizer.step()
            optimizer.zero_grad()
        reference_weight = reference.get_output_embeddings().weight
        if reference_engine is None:
            reference_weight = reference_weight[student_layout.start:student_layout.end]
        compare(head.weight, reference_weight,
                dtype, "optimizer head update")
        assert all(parameter.grad is None for parameter in teacher_engine.module.parameters())
        records.append({"case": "optimizer_update", "update": update + 1, "status": "PASS"})
    generate_and_compare("after_five_updates")
    result = {"status": "PASS", "scope": "Random tiny Qwen2 integration probe", "rank": rank,
              "dtype": args.dtype, "tied": args.tied, "records": records, "optimizer_updates": 5,
              "deepspeed_origin": deepspeed.__file__, "torch": torch.__version__,
              "reference_layout": "same AutoTP layout with dense loss" if reference_engine is not None else "unsharded FP32",
              "vocab_layout": {"vocab_size": student_layout.vocab_size, "start": student_layout.start,
                               "end": student_layout.end, "ranks": student_layout.ranks}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
