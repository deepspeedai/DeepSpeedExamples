# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Profiled steps as MeasuredStages: what BalancedTransformerBlocks(measured=...)
re-cuts the pipeline by."""

import json

from ray_deepspeed_pipeline.config import MASTER_WEIGHTS_KEY, MeasuredStages
from ray_deepspeed_pipeline.errors import ValidationError

_BUSY = ("forward", "backward", "vision_forward", "vision_backward")


def static_bytes_per_param(ds_config: dict, tp: int, dp: int) -> float:
    """Weights, gradients and optimizer state per GPU per parameter of a stage
    of tp x dp GPUs under its DeepSpeed config: bf16 weights; bf16 gradients,
    or an fp32 accumulation buffer; master weights, fp32 or 2 bytes (bf16, or
    rdsp's remainder); AdamW's two moments, fp32 or bf16. ZeRO 1 and 2 split
    the optimizer state over dp, ZeRO 2 the gradients too; under offload the
    optimizer state is in host memory. TP splits all of it."""
    bf16 = ds_config.get("bf16") or {}
    zero = ds_config.get("zero_optimization") or {}
    stage = int(zero.get("stage", 0) or 0)
    grads = 4.0 if (ds_config.get("data_types") or {}).get("grad_accum_dtype") == "fp32" else 2.0
    remainder = ds_config.get(MASTER_WEIGHTS_KEY) == "remainder"
    master = 2.0 if remainder or bf16.get("bf16_master_weights_and_grads") else 4.0
    moments = 4.0 if bf16.get("bf16_optimizer_states") and not remainder else 8.0
    optimizer = 0.0 if zero.get("offload_optimizer") else master + moments
    if stage >= 1:
        optimizer /= dp
    if stage >= 2:
        grads /= dp
    return (2.0 + grads + optimizer) / tp


def measured_stages(plan, records, skip: int, reserve_gib: float) -> MeasuredStages:
    """RayPipelineEngine.measure_stages' result from each rank's records
    (StageWorkerActor.profile_records): per stage, per rank."""
    seconds, peaks, totals = [], [], []
    for stage in records:
        steps = [[r for r in rank["records"] if r.get("kind") == "step"][skip:] for rank in stage]
        if not all(steps):
            raise ValidationError(
                "measure_stages: a rank recorded no step to measure; run more steps than skip"
            )
        per_step = [max(sum(r.get(k, 0.0) for k in _BUSY) for r in ranks) for ranks in zip(*steps)]
        seconds.append(sum(per_step) / len(per_step) / 1e3)
        # from the first kept step on, so each kept step's apply counts too
        kept = []
        for rank in stage:
            starts = [i for i, r in enumerate(rank["records"]) if r.get("kind") == "step"]
            kept += rank["records"][starts[skip] :]
        peaks.append(max((r.get("peak_gb", 0.0) for r in kept), default=0.0))
        totals += [rank["total_gib"] for rank in stage if rank["total_gib"] is not None]
    limit = (min(totals) if totals else float("inf")) - reserve_gib
    return MeasuredStages(
        cuts=tuple(s.block_start for s in plan.stages[1:]),
        seconds=tuple(seconds),
        peak_gib=tuple(peaks),
        bytes_per_param=tuple(
            static_bytes_per_param(json.loads(s.ds_config_json), s.tp, s.dp) for s in plan.stages
        ),
        limit_gib=limit,
    )
