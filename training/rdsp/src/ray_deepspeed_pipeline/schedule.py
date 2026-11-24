# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Deterministic 1F1B command generation from an ExecutionPlan.

Stage i runs min(stages-1-i, M) warm-up forwards, alternates one forward/one
backward, then drains the remaining backwards. Cross-stage dependencies are
explicit predecessor ids; intra-stage order is the sequence order. Tests check
the result against DeepSpeed's TrainSchedule.
"""

from ray_deepspeed_pipeline.plan import ExecutionPlan
from ray_deepspeed_pipeline.protocols import Command


def _cid(generation: int, step: int, stage: int, kind: str, microbatch: int | None) -> str:
    suffix = kind if microbatch is None else f"{kind}{microbatch}"
    return f"g{generation}.t{step}.s{stage}.{suffix}"


def _ops_1f1b(stage: int, n_stages: int, n_mb: int) -> list[tuple[str, int]]:
    warmup = min(n_stages - 1 - stage, n_mb)
    ops = [("forward", k) for k in range(warmup)]
    for k in range(n_mb - warmup):
        ops.append(("forward", warmup + k))
        ops.append(("backward", k))
    ops.extend(("backward", k) for k in range(n_mb - warmup, n_mb))
    return ops


def generate_commands(
    plan: ExecutionPlan, generation: int, global_step: int, train: bool = True
) -> dict[int, tuple[Command, ...]]:
    """Per-stage command sequences for one global step, in execution order.

    Train: 1F1B forwards/backwards, then a per-stage ready, then an apply that
    waits on every stage's ready (the all-stage barrier). Eval: forward-only
    "eval" commands, no ready/apply.
    """
    n_stages = len(plan.stages)
    n_mb = plan.global_microbatches

    def cid(stage, kind, microbatch):
        return _cid(generation, global_step, stage, kind, microbatch)

    def cmd(stage, kind, microbatch, preds):
        return Command(
            command_id=cid(stage, kind, microbatch),
            generation=generation,
            global_step=global_step,
            stage=stage,
            kind=kind,
            microbatch=microbatch,
            predecessors=tuple(preds),
        )

    out: dict[int, list[Command]] = {s: [] for s in range(n_stages)}

    if not train:
        for s in range(n_stages):
            for k in range(n_mb):
                preds = [cid(s - 1, "eval", k)] if s > 0 else []
                out[s].append(cmd(s, "eval", k, preds))
        return {s: tuple(cs) for s, cs in out.items()}

    for s in range(n_stages):
        for kind, k in _ops_1f1b(s, n_stages, n_mb):
            if kind == "forward":
                preds = [cid(s - 1, "forward", k)] if s > 0 else []
            elif s == n_stages - 1:
                preds = [cid(s, "forward", k)]  # the loss stage's own forward
            else:
                preds = [cid(s + 1, "backward", k)]
            out[s].append(cmd(s, kind, k, preds))

    ready_ids = []
    for s in range(n_stages):
        backward_ids = [c.command_id for c in out[s] if c.kind == "backward"]
        ready = cmd(s, "ready", None, backward_ids)
        out[s].append(ready)
        ready_ids.append(ready.command_id)
    for s in range(n_stages):
        out[s].append(cmd(s, "apply", None, ready_ids))

    return {s: tuple(cs) for s, cs in out.items()}


def peak_in_flight(sequence) -> int:
    """Most forward activations one stage's op order holds at once."""
    live = peak = 0
    for c in sequence:
        if c.kind == "forward":
            live += 1
            peak = max(peak, live)
        elif c.kind == "backward":
            live -= 1
    return peak


def control_commands(
    plan: ExecutionPlan, generation: int, global_step: int, kind: str
) -> dict[int, Command]:
    """One save/load command per stage, sent to every rank of the stage
    (DeepSpeed's save/load are stage-local collectives)."""
    assert kind in ("save", "load"), kind
    return {
        s: Command(
            command_id=_cid(generation, global_step, s, kind, None),
            generation=generation,
            global_step=global_step,
            stage=s,
            kind=kind,
            microbatch=None,
            predecessors=(),
        )
        for s in range(len(plan.stages))
    }
