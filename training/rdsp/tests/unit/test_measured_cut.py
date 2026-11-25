# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""--cuts measured: profiled stage times re-cut the pipeline. Each stage's
time beyond its blocks' becomes a fixed cost of that stage, blocks move off
slow stages, and no stage is cut to need more memory than the limit."""

import json
import types

import pytest

from ray_deepspeed_pipeline.compiler import _recompute_blocks
from ray_deepspeed_pipeline.config import (
    BalancedTransformerBlocks,
    MeasuredStages,
    StageOverride,
)
from ray_deepspeed_pipeline.errors import ValidationError
from ray_deepspeed_pipeline.measurement import measured_stages, static_bytes_per_param
from ray_deepspeed_pipeline.partition import measured_cuts, measured_extras, partition_parameters

BLOCK = 100  # parameters per block


def measured(cuts, seconds, peak=(10.0, 10.0), bpp=(1.0, 1.0), limit=80.0):
    return MeasuredStages(
        cuts=cuts, seconds=seconds, peak_gib=peak, bytes_per_param=bpp, limit_gib=limit
    )


def costs(n=32, post=300):
    return [BLOCK] * n, post


def test_the_first_stage_s_unexplained_time_becomes_its_fixed_cost():
    # 16 | 16 blocks, the last stage's head 3 blocks' worth: 19 units take 19 s,
    # so the first stage's 25 s is 16 blocks and 9 blocks' worth of extra
    extras = measured_extras(*costs(), (1, 1), measured((16,), (25.0, 19.0)))
    assert extras == pytest.approx([9 * BLOCK, 3 * BLOCK])


def test_blocks_move_off_the_slow_stage():
    # 16 + 9 against 16 + 3: 6 blocks too many on the first stage, so 3 move
    assert measured_cuts(*costs(), (1, 1), 0, measured((16,), (25.0, 19.0))) == [13]


def test_times_that_match_the_estimate_give_the_estimate_s_cut():
    from ray_deepspeed_pipeline.partition import _balanced_cuts

    # 16 blocks in 16 s, 16 blocks and the head (3 blocks' worth) in 19 s
    assert (
        measured_cuts(*costs(), (1, 1), 0, measured((16,), (16.0, 19.0)))
        == _balanced_cuts((0, *costs()), 2, 0, (1, 1))
        == [18]
    )


def test_middle_stages_set_the_rate():
    # 3 stages of 8 blocks: the middle one takes 8 s, so 1 s per block; the
    # first carries 4 blocks' worth of extra, the last 2: 10 per stage is 6 | 10 | 8
    m = MeasuredStages(
        cuts=(8, 16),
        seconds=(12.0, 8.0, 10.0),
        peak_gib=(1, 1, 1),
        bytes_per_param=(1, 1, 1),
        limit_gib=80,
    )
    assert measured_extras(*costs(24, post=0), (1, 1, 1), m) == pytest.approx(
        [4 * BLOCK, 0, 2 * BLOCK]
    )
    assert measured_cuts(*costs(24, post=0), (1, 1, 1), 0, m) == [6, 16]


def test_memory_caps_what_a_stage_gains():
    # each block is 1 GiB on the last stage, which has 2 GiB to spare: it takes
    # two blocks, not three
    m = measured((16,), (25.0, 19.0), peak=(70.0, 74.0), bpp=(1.0, 2**30 / BLOCK), limit=76.0)
    assert measured_cuts(*costs(), (1, 1), 0, m) == [14]
    full = measured((16,), (25.0, 19.0), peak=(70.0, 76.0), bpp=(1.0, 2**30 / BLOCK), limit=76.0)
    assert measured_cuts(*costs(), (1, 1), 0, full) == [16]


def test_a_stage_already_over_the_limit_may_still_lose_blocks():
    m = measured((16,), (19.0, 25.0), peak=(70.0, 79.0), bpp=(1.0, 1.0), limit=76.0)
    assert measured_cuts(*costs(), (1, 1), 0, m)[0] > 16


def test_through_partition_parameters():
    from test_partition import ToyLM

    model = ToyLM(n_blocks=8)
    cut = partition_parameters(model, BalancedTransformerBlocks(), 2)[0].block_stop
    slow_first = BalancedTransformerBlocks(measured=measured((cut,), (9.0, 4.5)))
    assert partition_parameters(model, slow_first, 2)[0].block_stop < cut
    with pytest.raises(ValidationError, match="do not fit"):
        partition_parameters(
            model, BalancedTransformerBlocks(measured=measured((9,), (1.0, 1.0))), 2
        )


@pytest.mark.parametrize(
    "conf,tp,dp,expected",
    [
        # bf16 weights 2 + bf16 grads 2 + (fp32 master 4 + moments 8) / dp 2
        ({"zero_optimization": {"stage": 1}}, 2, 2, (2 + 2 + 12 / 2) / 2),
        # rdsp's remainder: a 2-byte master
        (
            {"zero_optimization": {"stage": 1}, "rdsp_master_weights": "remainder"},
            1,
            2,
            2 + 2 + 10 / 2,
        ),
        # fp32 gradient accumulation, no ZeRO
        ({"data_types": {"grad_accum_dtype": "fp32"}}, 1, 4, 2 + 4 + 12),
        # ZeRO-2 splits the gradients too; bf16 master and moments
        (
            {
                "zero_optimization": {"stage": 2},
                "bf16": {"bf16_master_weights_and_grads": True, "bf16_optimizer_states": True},
            },
            1,
            2,
            2 + 2 / 2 + 6 / 2,
        ),
        # offload: no optimizer state on the GPU
        ({"zero_optimization": {"stage": 1, "offload_optimizer": {"device": "cpu"}}}, 1, 2, 4),
    ],
)
def test_static_bytes_per_param(conf, tp, dp, expected):
    assert static_bytes_per_param(conf, tp, dp) == pytest.approx(expected)


def rank(*steps, peak=50.0, total=79.0):
    records = []
    for fwd, bwd in steps:
        records.append(
            {
                "stage": 0,
                "kind": "step",
                "forward": fwd,
                "backward": bwd,
                "wait_bwd": 999.0,
                "peak_gb": peak,
            }
        )
        records.append({"stage": 0, "kind": "apply", "apply": 50.0, "peak_gb": peak + 1})
    return {"records": records, "total_gib": total}


def test_measured_stages_from_rank_records():
    conf = json.dumps({"zero_optimization": {"stage": 1}})
    spec = dict(ds_config_json=conf, tp=2, dp=2)
    plan = types.SimpleNamespace(
        stages=[
            types.SimpleNamespace(block_start=0, **spec),
            types.SimpleNamespace(block_start=30, **spec),
        ]
    )
    records = [
        # stage 0: its slowest rank per step, the first step skipped, waits not counted
        [
            rank((9000, 9000), (5000, 11000), (5000, 11400)),
            rank((1, 1), (6000, 11000), (1, 1), peak=70.0),
        ],
        [rank((1, 1), (4000, 8000), (4000, 8200), total=80.0)],
    ]
    m = measured_stages(plan, records, skip=1, reserve_gib=5.0)
    assert m.cuts == (30,)
    assert m.seconds == pytest.approx(((17000 + 16400) / 2 / 1e3, (12000 + 12200) / 2 / 1e3))
    assert m.peak_gib == (71.0, 51.0)  # the apply's peak counts
    assert m.limit_gib == 74.0
    assert m.bytes_per_param == pytest.approx(((2 + 2 + 6) / 2,) * 2)


def test_recompute_blocks_follows_the_blocks_a_stage_gains():
    policy = BalancedTransformerBlocks(measured=measured((32,), (1.0, 1.0)))
    part0 = types.SimpleNamespace(index=0, block_start=0, block_stop=28)
    part1 = types.SimpleNamespace(index=1, block_start=28, block_stop=64)
    over = StageOverride(stage=0, recompute_blocks=24)
    # stage 0 gave up 4 blocks: 4 fewer recomputed; stage 1 gained 4: 4 more
    assert _recompute_blocks(over, part0, policy, 64) == 20
    assert _recompute_blocks(StageOverride(stage=1, recompute_blocks=24), part1, policy, 64) == 28
    assert _recompute_blocks(over, part0, BalancedTransformerBlocks(), 64) == 24
    assert _recompute_blocks(None, part0, policy, 64) is None


def tiny_qwen3_5_vl(layers):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3_5ForConditionalGeneration"):
        pytest.skip("transformers without Qwen3.5")
    cfg = transformers.Qwen3_5Config(
        text_config=dict(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            layer_types=["linear_attention", "linear_attention", "full_attention"] * (layers // 3),
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
                "partial_rotary_factor": 0.5,
            },
        ),
        vision_config=dict(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=64,
            patch_size=4,
            spatial_merge_size=2,
            temporal_patch_size=1,
            num_position_embeddings=16,
        ),
        image_token_id=250,
        video_token_id=251,
        vision_start_token_id=252,
    )
    return transformers.Qwen3_5ForConditionalGeneration(cfg)


def test_a_vision_stage_keeps_its_cut_and_the_decoder_stages_are_re_cut():
    """--cuts 0,measured: the vision encoder alone on stage 0 (2 GPUs), the decoder's 12
    blocks over three 1-GPU stages. The last stage measured slow (its head): blocks move
    off it to the other decoder stages; stage 0 keeps no blocks."""
    import dataclasses

    model = tiny_qwen3_5_vl(layers=12)
    policy = BalancedTransformerBlocks(vision_stage=True)
    gpus = (2, 1, 1, 1)
    before = partition_parameters(model, policy, 4, stage_gpus=gpus)
    cuts = tuple(p.block_stop for p in before[:-1])
    assert cuts[0] == 0
    m = MeasuredStages(
        cuts=cuts,
        seconds=(5.0, 4.0, 4.0, 9.0),
        peak_gib=(10.0,) * 4,
        bytes_per_param=(1.0,) * 4,
        limit_gib=80.0,
    )
    after = partition_parameters(model, dataclasses.replace(policy, measured=m), 4, stage_gpus=gpus)
    blocks = [p.block_stop - p.block_start for p in after]
    assert blocks[0] == 0 and sum(blocks) == 12
    assert blocks[3] < before[3].block_stop - before[3].block_start


def test_a_vision_stage_needs_measurements_with_its_cut():
    model = tiny_qwen3_5_vl(layers=12)
    m = MeasuredStages(
        cuts=(4, 8),
        seconds=(1.0, 1.0, 1.0),
        peak_gib=(10.0,) * 3,
        bytes_per_param=(1.0,) * 3,
        limit_gib=80.0,
    )
    with pytest.raises(ValidationError, match="cut at 0"):
        partition_parameters(model, BalancedTransformerBlocks(vision_stage=True, measured=m), 3)


def test_a_failed_step_is_what_measure_stages_raises_even_if_profiling_cannot_be_switched_off():
    from ray_deepspeed_pipeline.engine import RayPipelineEngine
    from ray_deepspeed_pipeline.errors import StepFailed

    class Coordinator:  # a rank died mid-step: its actor no longer answers
        def set_profile(self, on):
            if not on:
                raise RuntimeError("actor died")

        def train_batch(self, data_iter):
            raise StepFailed("stage 1 rank 0 was lost")

    with pytest.raises(StepFailed):
        RayPipelineEngine(Coordinator()).measure_stages(iter([]))


def test_stage_times_must_be_positive():
    with pytest.raises(ValidationError, match="positive"):
        measured((16,), (0.0, 19.0))


def test_skipped_steps_set_no_memory_peak():
    conf = json.dumps({"zero_optimization": {"stage": 1}})
    plan = types.SimpleNamespace(
        stages=[types.SimpleNamespace(block_start=0, ds_config_json=conf, tp=1, dp=1)]
    )
    warm_up = rank((9000, 9000), peak=78.0)["records"]  # allocation, compilation
    steady = rank((5000, 11000), (5000, 11000), peak=50.0)["records"]
    records = [[{"records": warm_up + steady, "total_gib": 80.0}]]
    m = measured_stages(plan, records, skip=1, reserve_gib=5.0)
    assert m.peak_gib == (51.0,)
