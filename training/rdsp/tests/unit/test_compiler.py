# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""The single lowering path from PipelineConfig to ExecutionPlan.

Test names double as acceptance selectors; keep them stable."""

import dataclasses
import json

import pytest
from test_partition import ToyLM

from ray_deepspeed_pipeline.compiler import lower, resolve_microbatches
from ray_deepspeed_pipeline.config import (
    ConnectionOverride,
    ExplicitCuts,
    PipelineConfig,
    StageOverride,
    UniformTransformerBlocks,
)
from ray_deepspeed_pipeline.errors import ValidationError

DS = {"train_batch_size": 8, "gradient_accumulation_steps": 4}


def simple_config(**kw):
    return PipelineConfig(stages=2, partition=UniformTransformerBlocks(), **kw)


def test_simple_complete_plan():
    plan = lower(ToyLM(), simple_config(), DS)
    assert len(plan.stages) == 2 and len(plan.connections) == 1
    assert plan.global_microbatches == 4
    assert plan.schedule.kind == "1f1b"
    assert all(s.num_gpus == 1 for s in plan.stages)
    assert plan.connections[0].conversion == "identity"
    # stage ds config stays an ordinary DeepSpeed config
    assert json.loads(plan.stages[0].ds_config_json)["train_batch_size"] == 8


def test_microbatches_from_gas():
    assert resolve_microbatches(simple_config(), DS) == 4
    assert resolve_microbatches(simple_config(microbatches=8), {}) == 8
    with pytest.raises(ValidationError):
        resolve_microbatches(simple_config(), {})  # neither source


def test_mismatch_rejected():
    with pytest.raises(ValidationError, match="microbatches"):
        lower(ToyLM(), simple_config(microbatches=8), DS)


def test_equal_explicit_value_same_plan_hash():
    implicit = lower(ToyLM(), simple_config(), DS)
    explicit = lower(ToyLM(), simple_config(microbatches=4), DS)
    assert implicit.plan_hash() == explicit.plan_hash()


def test_simple_advanced_same_plan():
    # an advanced config that spells out the defaults lowers byte-identically
    simple = lower(ToyLM(), simple_config(), DS)
    advanced = lower(
        ToyLM(),
        simple_config(
            stage_overrides=(
                StageOverride(stage=0, num_gpus=1),
                StageOverride(stage=1, num_gpus=1),
            ),
            connection_overrides=(ConnectionOverride(source=0, dest=1),),
        ),
        DS,
    )
    assert simple.canonical_json() == advanced.canonical_json()
    assert simple.plan_hash() == advanced.plan_hash()


def test_heterogeneous_one_global_count():
    ds = {"train_batch_size": 16, "gradient_accumulation_steps": 4}  # 4 rows/mb
    plan = lower(
        ToyLM(),
        simple_config(
            stage_overrides=(
                StageOverride(stage=0, num_gpus=4, zero_stage=1),
                StageOverride(stage=1, num_gpus=2),
            ),
        ),
        ds,
    )
    assert [s.num_gpus for s in plan.stages] == [4, 2]
    assert [s.rows_per_rank for s in plan.stages] == [1, 2]
    assert plan.connections[0].conversion == "shard-to-shard"
    assert json.loads(plan.stages[0].ds_config_json)["zero_optimization"]["stage"] == 1
    # differing stage-local resources never fork the global microbatch count
    assert plan.global_microbatches == plan.schedule.global_microbatches == 4


def test_unsupported_layout_rejected_before_ray():
    with pytest.raises(ValidationError, match="layouts require 'identity'"):
        lower(
            ToyLM(),
            simple_config(
                connection_overrides=(
                    ConnectionOverride(source=0, dest=1, conversion="replicate-to-shard"),
                )
            ),
            DS,
        )
    with pytest.raises(ValidationError, match="adjacent"):
        lower(
            ToyLM(),
            PipelineConfig(
                stages=3,
                partition=UniformTransformerBlocks(),
                connection_overrides=(ConnectionOverride(source=2, dest=3),),
            ),
            DS,
        )


def test_duplicate_stage_override_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        lower(
            ToyLM(),
            simple_config(stage_overrides=(StageOverride(stage=0), StageOverride(stage=0))),
            DS,
        )


def test_plan_deterministic_across_model_instances():
    # partitioning is name-based, so two identically-shaped models lower to
    # the same plan hash (weights don't enter the plan)
    assert (
        lower(ToyLM(), simple_config(), DS).plan_hash()
        == lower(ToyLM(), simple_config(), DS).plan_hash()
    )


def test_per_stage_gradient_clipping_never_silently_on():
    plan = lower(ToyLM(), simple_config(), DS)
    # DeepSpeed's own default (1.0) would clip per stage: forced off
    assert all(json.loads(st.ds_config_json)["gradient_clipping"] == 0.0 for st in plan.stages)
    with pytest.raises(ValidationError, match="global"):
        lower(ToyLM(), simple_config(), dict(DS, gradient_clipping=1.0))


def test_optimizer_offload_only_on_the_stage_that_asks():
    plan = lower(
        ToyLM(),
        simple_config(
            stage_overrides=(StageOverride(stage=0, zero_stage=2, offload_optimizer=True),)
        ),
        DS,
    )
    zero0 = json.loads(plan.stages[0].ds_config_json)["zero_optimization"]
    assert zero0["offload_optimizer"] == {"device": "cpu", "pin_memory": True}
    assert "offload_optimizer" not in json.loads(plan.stages[1].ds_config_json).get(
        "zero_optimization", {}
    )


def test_optimizer_offload_inherits_zero_from_the_config():
    plan = lower(
        ToyLM(),
        simple_config(stage_overrides=(StageOverride(stage=1, offload_optimizer=True),)),
        {**DS, "zero_optimization": {"stage": 1}},
    )
    assert json.loads(plan.stages[1].ds_config_json)["zero_optimization"]["stage"] == 1


def test_optimizer_offload_without_zero_rejected():
    with pytest.raises(ValidationError, match="ZeRO stage 1 or 2"):
        lower(
            ToyLM(),
            simple_config(stage_overrides=(StageOverride(stage=0, offload_optimizer=True),)),
            DS,
        )


def test_balanced_cuts_give_a_multi_gpu_stage_more_blocks():
    """ToyLM blocks cost 72, the norm and head 176. With the middle stage on 2
    GPUs its cost counts half: every split keeping all stages within 248 (the
    head stage's minimum) is 1|4|1, 2|3|1 or 3|2|1, and 1|4|1 (72, 144/2, 248)
    spreads the rest most evenly. On single-GPU stages it would be 3|2|1."""
    from ray_deepspeed_pipeline.config import BalancedTransformerBlocks

    plan = lower(
        ToyLM(n_blocks=6),
        PipelineConfig(
            stages=3,
            partition=BalancedTransformerBlocks(),
            stage_overrides=(StageOverride(stage=1, num_gpus=2),),
        ),
        DS,
    )
    assert [(s.block_start, s.block_stop) for s in plan.stages] == [(0, 1), (1, 5), (5, 6)]


def test_prefetch_is_off_unless_asked_and_reaches_the_plan():
    assert not lower(ToyLM(), simple_config(), DS).prefetch
    cfg = dataclasses.replace(simple_config(), prefetch=True)
    assert lower(ToyLM(), cfg, DS).prefetch


def test_compile_is_per_stage():
    cfg = simple_config(stage_overrides=(StageOverride(stage=1, compile=True),))
    assert [s.compile for s in lower(ToyLM(), cfg, DS).stages] == [False, True]


def test_compile_vision_is_per_stage():
    cfg = simple_config(stage_overrides=(StageOverride(stage=0, compile_vision=True),))
    assert [s.compile_vision for s in lower(ToyLM(), cfg, DS).stages] == [True, False]


def test_gradient_clipping_is_rejected_for_one_stage_too():
    """One stage would otherwise force clipping off silently."""
    one = PipelineConfig(stages=1, partition=UniformTransformerBlocks())
    with pytest.raises(ValidationError, match="gradient_clipping"):
        lower(ToyLM(), one, dict(DS, gradient_clipping=1.0))


def test_settings_that_do_not_change_checkpoints_keep_the_plan_hash():
    """A checkpoint is bound to the plan hash; prefetch, recompute and
    compile change how steps run, not what a stage saves."""
    base = lower(ToyLM(), simple_config(), DS).plan_hash()
    runtime = simple_config(
        prefetch=True,
        stage_overrides=(
            StageOverride(stage=0, recompute=True, compile=True, compile_vision=True),
        ),
    )
    assert lower(ToyLM(), runtime, DS).plan_hash() == base
    moved = PipelineConfig(stages=2, partition=ExplicitCuts((1,)))
    assert lower(ToyLM(), moved, DS).plan_hash() != base


def test_recompute_blocks_reaches_the_stage_spec_and_excludes_recompute():
    plan = lower(
        ToyLM(), simple_config(stage_overrides=(StageOverride(stage=0, recompute_blocks=3),)), DS
    )
    assert plan.stages[0].recompute_blocks == 3 and not plan.stages[0].recompute
    for bad, match in (
        (dict(recompute=True, recompute_blocks=2), "set one"),
        (dict(recompute_blocks=-1), ">= 0"),
    ):
        with pytest.raises(ValidationError, match=match):
            lower(ToyLM(), simple_config(stage_overrides=(StageOverride(stage=0, **bad),)), DS)


REMAINDER = {
    **DS,
    "bf16": {"enabled": True},
    "zero_optimization": {"stage": 1},
    "rdsp_master_weights": "remainder",
    "optimizer": {"type": "AdamW", "params": {"lr": 1e-5, "betas": [0.9, 0.95]}},
}


def test_remainder_master_config_lowers():
    plan = lower(ToyLM(), simple_config(), REMAINDER)
    assert json.loads(plan.stages[0].ds_config_json)["rdsp_master_weights"] == "remainder"


@pytest.mark.parametrize(
    "change,needs",
    [
        ({"rdsp_master_weights": "remaindr"}, "'remainder'"),
        ({"bf16": {"enabled": False}}, "bf16"),
        ({"zero_optimization": {"stage": 0}}, "ZeRO stage 1 or 2"),
        ({"zero_optimization": {"stage": 3}}, "ZeRO stage 1 or 2"),
        ({"data_types": {"grad_accum_dtype": "fp32"}}, "bf16 gradient accumulation"),
        ({"optimizer": {"type": "SGD", "params": {"lr": 0.1}}}, "AdamW"),
        ({"optimizer": {"type": "AdamW", "params": {"adam_w_mode": True}}}, "adam_w_mode"),
    ],
)
def test_remainder_master_misuse_rejected_before_ray(change, needs):
    # the stage adapter would ignore the key, swap the optimizer, or fail inside an actor
    with pytest.raises(ValidationError, match=needs):
        lower(ToyLM(), simple_config(), {**REMAINDER, **change})


def test_remainder_master_checked_per_stage():
    offload = StageOverride(stage=1, zero_stage=1, offload_optimizer=True)
    with pytest.raises(ValidationError, match="stage 1.*offload"):
        lower(ToyLM(), simple_config(stage_overrides=(offload,)), REMAINDER)
    with pytest.raises(ValidationError, match="stage 1.*ZeRO stage 1 or 2"):
        lower(
            ToyLM(),
            simple_config(stage_overrides=(StageOverride(stage=1, zero_stage=0),)),
            REMAINDER,
        )


@pytest.mark.parametrize("value", [-1, "many", 2.5])
def test_optimizer_group_size_must_be_a_whole_number(value):
    with pytest.raises(ValidationError, match="rdsp_optimizer_group_numel"):
        lower(ToyLM(), simple_config(), {**DS, "rdsp_optimizer_group_numel": value})
    lower(ToyLM(), simple_config(), {**DS, "rdsp_optimizer_group_numel": 0})


def test_fp16_is_rejected_for_every_pipeline():
    # each stage's engine.backward would scale by its own loss scale, so upstream
    # gradients would carry every downstream stage's scale too
    with pytest.raises(ValidationError, match="fp16.*bf16 or fp32"):
        lower(ToyLM(), simple_config(), {**DS, "fp16": {"enabled": True}})
    lower(ToyLM(), simple_config(), {**DS, "fp16": {"enabled": False}})


def test_sequence_parallelism_is_rejected_for_linear_attention_layers():
    import types

    def model(layer_types):
        m = ToyLM()
        m.config = types.SimpleNamespace(layer_types=layer_types)
        return m

    cfg = simple_config(stage_overrides=[StageOverride(stage=0, num_gpus=2, sp=2)])
    with pytest.raises(ValidationError, match="linear.attention"):
        lower(model(["linear_attention", "full_attention"]), cfg, DS)
    assert lower(model(["full_attention"] * 2), cfg, DS).stages[0].sp == 2
