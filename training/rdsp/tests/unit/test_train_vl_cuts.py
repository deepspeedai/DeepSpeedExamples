# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""train_vl.py --cuts: each spelling maps to one partition policy."""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

import ray_deepspeed_pipeline as rdsp  # noqa: E402
import train_vl  # noqa: E402


def test_balanced_measures_the_vision_ratio_from_the_first_step():
    policy = train_vl.partition_policy("balanced", ratio=lambda: 2.5)
    assert policy == rdsp.BalancedTransformerBlocks(vision_token_ratio=2.5)


@pytest.mark.parametrize("cuts", ["0,balanced", "0,measured"])
def test_zero_then_balanced_gives_the_encoder_its_own_stage(cuts):
    # 0,measured calibrates with this cut, then re-cuts the decoder stages
    policy = train_vl.partition_policy(cuts, ratio=lambda: 1 / 0)
    assert policy == rdsp.BalancedTransformerBlocks(vision_stage=True)


@pytest.mark.parametrize(
    "cuts,policy",
    [
        ("even", rdsp.UniformTransformerBlocks()),
        ("0,16", rdsp.ExplicitCuts((0, 16))),
        ("13", rdsp.ExplicitCuts((13,))),
    ],
)
def test_even_and_explicit_cuts(cuts, policy):
    assert train_vl.partition_policy(cuts, ratio=lambda: 1 / 0) == policy


def test_measured_calibrates_with_the_balanced_cut():
    policy = train_vl.partition_policy("measured", ratio=lambda: 2.5)
    assert policy == rdsp.BalancedTransformerBlocks(vision_token_ratio=2.5)


def test_calibration_measures_at_learning_rate_zero_and_shuts_down():
    """calibrated_policy: profiled steps on the run's own first batches, the
    engine shut down after, the estimate re-cut by what it measured."""
    args = train_vl.build_parser().parse_args(
        [
            "--cuts",
            "measured",
            "--calibration-steps",
            "3",
            "--microbatches",
            "2",
            "--drop-padding-mask",
        ]
    )
    assert train_vl.ds_config(args, lr=0.0)["optimizer"]["params"]["lr"] == 0.0
    assert train_vl.ds_config(args)["optimizer"]["params"]["lr"] == args.lr

    measured = rdsp.MeasuredStages(
        cuts=(3,), seconds=(2.0, 1.0), peak_gib=(1, 1), bytes_per_param=(1, 1), limit_gib=70
    )

    class Engine:
        shut = False

        def measure_stages(self, data_iter, steps, skip):
            self.read = list(data_iter)
            return measured

        def shutdown(self):
            self.shut = True

    import torch

    data = iter(
        [
            ({"input_ids": torch.ones(1, 4), "attention_mask": torch.ones(1, 4)}, i)
            for i in range(100)
        ]
    )
    engine = Engine()
    estimate = rdsp.BalancedTransformerBlocks(vision_token_ratio=2.5)
    policy = train_vl.calibrated_policy(args, engine, data, estimate)
    assert policy == rdsp.BalancedTransformerBlocks(vision_token_ratio=2.5, measured=measured)
    assert engine.shut
    assert [y for _, y in engine.read] == list(range(8))  # 3 steps and one read ahead
    assert all("attention_mask" not in x for x, _ in engine.read)


@pytest.mark.parametrize("flags", [["--calibration-steps", "1"], ["--optimizer-group-numel", "-5"]])
def test_out_of_range_numbers_are_refused_when_parsed(flags, capsys):
    # one calibration step leaves none to measure (the first is not counted)
    with pytest.raises(SystemExit):
        train_vl.build_parser().parse_args(flags)
    assert flags[0] in capsys.readouterr().err
