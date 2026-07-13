# DeepSpeed Team
"""Distributed helpers for tensor-parallel-sharded logits.

Tensor parallelism (TP > 1) makes the LM head column-parallel, so a forward
returns logits sharded along the vocab dim as ``[B, T, V/N]`` instead of the
full ``[B, T, V]`` distribution that the distillation loss and the teacher
logit cache expect. :func:`gather_vocab_if_sharded` stitches them back into
the full distribution.

ZeRO-3 (parameter partitioning) does *not* shard the vocab dim -- every rank
recomputes the full-V logits after gathering parameters -- so the gather is a
no-op there. The two are told apart by comparing the tensor's last dim against
the model's real ``vocab_size``, never by ``world_size`` (which is > 1 under
both strategies).

``deepspeed.comm`` is imported lazily inside the function so that importing
this module stays free of distributed dependencies; the CPU-only unit tests
rely on that.
"""

import torch


def gather_vocab_if_sharded(logits: torch.Tensor,
                            full_vocab_size: int,
                            group=None) -> torch.Tensor:
    """All-gather ``logits`` along the vocab dim if TP left them sharded.

    Self-detects via shape: when ``logits.shape[-1] == full_vocab_size`` the
    tensor already holds the full distribution (TP == 1, or ZeRO-3) and is
    returned untouched. Only when the last dim is smaller do we all-gather
    across ``group`` (default: world) and concatenate along ``dim=-1``.

    Args:
        logits: ``[B, T, *]`` tensor from a model forward. Under TP it is
            ``[B, T, V/N]``; otherwise ``[B, T, V]``.
        full_vocab_size: the model's real ``config.vocab_size``. Used as the
            "already complete" threshold, so the call is a no-op for TP == 1
            and for ZeRO-3.
        group: optional process group for the gather. Defaults to the world
            group, which is correct when the model is TP-only (no data
            parallelism). Pass the TP group explicitly if DP and TP are
            combined so only the TP dimension is gathered.
    """
    if logits.shape[-1] == full_vocab_size:
        return logits

    from deepspeed.comm import dist

    if not dist.is_initialized():
        raise RuntimeError(
            f"logits vocab dim ({logits.shape[-1]}) < full vocab size "
            f"({full_vocab_size}), which means the LM head is TP-sharded, "
            "but torch.distributed is not initialized.")

    world = dist.get_world_size(group)
    shards = [torch.empty_like(logits) for _ in range(world)]
    dist.all_gather(shards, logits.contiguous(), group=group)
    gathered = torch.cat(shards, dim=-1)

    if gathered.shape[-1] != full_vocab_size:
        raise RuntimeError(
            f"gathered vocab dim ({gathered.shape[-1]}) != full vocab size "
            f"({full_vocab_size}); world_size={world} likely does not match "
            "the tensor-parallel degree.")
    return gathered
