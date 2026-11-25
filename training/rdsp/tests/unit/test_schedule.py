# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Deterministic 1F1B command generation."""

from test_partition import ToyLM

from ray_deepspeed_pipeline.compiler import lower
from ray_deepspeed_pipeline.config import PipelineConfig, UniformTransformerBlocks
from ray_deepspeed_pipeline.schedule import generate_commands, peak_in_flight


def make_plan(stages=2, microbatches=4):
    cfg = PipelineConfig(
        stages=stages, partition=UniformTransformerBlocks(), microbatches=microbatches
    )
    return lower(ToyLM(), cfg, {})  # schedule shape only; 1 row per microbatch


def ops(sequence):
    return [(c.kind, c.microbatch) for c in sequence if c.kind in ("forward", "backward")]


def test_deterministic_1f1b():
    plan = make_plan()
    a = generate_commands(plan, 1, 0)
    b = generate_commands(plan, 1, 0)
    assert a == b
    ids = [c.command_id for seq in a.values() for c in seq]
    assert len(ids) == len(set(ids)), "command ids must be unique"


def test_deepspeed_pipeline_oracle():
    """2-stage / 4-microbatch golden trace matching pinned TrainSchedule
    semantics: last stage strictly alternates F/B; first stage runs one
    warm-up forward then alternates."""
    seqs = generate_commands(make_plan(2, 4), 1, 0)
    F, B = "forward", "backward"
    assert ops(seqs[0]) == [(F, 0), (F, 1), (B, 0), (F, 2), (B, 1), (F, 3), (B, 2), (B, 3)]
    assert ops(seqs[1]) == [(F, 0), (B, 0), (F, 1), (B, 1), (F, 2), (B, 2), (F, 3), (B, 3)]


def test_bounded_activation_lifetime():
    for stages, mbs in [(2, 4), (3, 6), (4, 8), (4, 2)]:
        seqs = generate_commands(make_plan(stages, mbs), 1, 0)
        for s in range(stages):
            assert peak_in_flight(seqs[s]) <= min(stages - s, mbs), (
                f"stage {s} of {stages} exceeds 1F1B's activation bound"
            )


def test_cross_stage_predecessors():
    seqs = generate_commands(make_plan(3, 2), 7, 5)
    by_id = {c.command_id: c for seq in seqs.values() for c in seq}
    # forward chains left to right
    f_s1 = next(c for c in seqs[1] if c.kind == "forward" and c.microbatch == 0)
    assert f_s1.predecessors == ("g7.t5.s0.forward0",)
    # backward chains right to left; terminal backward depends on own forward
    b_s0 = next(c for c in seqs[0] if c.kind == "backward" and c.microbatch == 0)
    assert b_s0.predecessors == ("g7.t5.s1.backward0",)
    b_term = next(c for c in seqs[2] if c.kind == "backward" and c.microbatch == 0)
    assert b_term.predecessors == ("g7.t5.s2.forward0",)
    # apply is gated on EVERY stage's ready (the all-stage barrier)
    apply_s0 = next(c for c in seqs[0] if c.kind == "apply")
    assert set(apply_s0.predecessors) == {f"g7.t5.s{s}.ready" for s in range(3)}
    assert all(p in by_id for c in by_id.values() for p in c.predecessors)


def test_eval_schedule_is_forward_only_with_no_apply():
    seqs = generate_commands(make_plan(2, 4), 1, 0, train=False)
    kinds = {c.kind for seq in seqs.values() for c in seq}
    assert kinds == {"eval"}
