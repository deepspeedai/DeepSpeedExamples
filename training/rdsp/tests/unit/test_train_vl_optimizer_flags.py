# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""train_vl.py optimizer flags. --master-dtype bf16 sets DeepSpeed's two
low-precision state keys and is refused wherever DeepSpeed would ignore them;
--master-dtype remainder asks the stage adapter for RemainderAdamW under the
same conditions; --fused-adam asks PyTorch's AdamW for its fused kernel."""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

import train_vl  # noqa: E402

STAGES = ["--stages", "2", "--stage", "0:gpus=4,tp=4,zero=1", "--stage", "1:gpus=4,tp=4,zero=1"]


def parse(*flags):
    return train_vl.build_parser().parse_args([*STAGES, *flags])


def test_fp32_is_the_default_and_adds_nothing():
    args = parse()
    assert args.master_dtype == "fp32"
    bf16 = train_vl.ds_config(args)["bf16"]
    assert "bf16_master_weights_and_grads" not in bf16 and "bf16_optimizer_states" not in bf16
    assert train_vl.optimizer_problem(args) is None


def test_bf16_sets_both_keys_and_leaves_grads_bf16():
    args = parse("--master-dtype", "bf16", "--grad-dtype", "bf16")
    conf = train_vl.ds_config(args)
    assert conf["bf16"]["bf16_master_weights_and_grads"] is True
    assert conf["bf16"]["bf16_optimizer_states"] is True
    assert "data_types" not in conf  # gradients accumulate in the model's bf16
    assert train_vl.optimizer_problem(args) is None


def test_remainder_asks_the_adapter_and_leaves_deepspeed_s_keys_alone():
    args = parse("--master-dtype", "remainder", "--grad-dtype", "bf16")
    conf = train_vl.ds_config(args)
    assert conf["rdsp_master_weights"] == "remainder"
    assert "bf16_master_weights_and_grads" not in conf["bf16"]
    assert "data_types" not in conf
    assert train_vl.optimizer_problem(args) is None
    assert "rdsp_master_weights" not in train_vl.ds_config(parse())


@pytest.mark.parametrize("master", ["bf16", "remainder"])
@pytest.mark.parametrize(
    "flags,needs",
    [
        ([], "--grad-dtype bf16"),
        (["--grad-dtype", "bf16", "--colocated-vision"], "replicated"),
    ],
)
def test_low_precision_masters_refused_where_deepspeed_would_keep_fp32(master, flags, needs):
    problem = train_vl.optimizer_problem(parse("--master-dtype", master, *flags))
    assert needs in problem and f"--master-dtype {master}" in problem


@pytest.mark.parametrize("master", ["bf16", "remainder"])
@pytest.mark.parametrize(
    "stage,needs",
    [
        ("1:gpus=4,tp=4,zero=0", "stage 1 has 0"),
        ("1:gpus=4,tp=4", "stage 1 has 0"),  # inherits --zero's default 0
        ("1:gpus=4,tp=4,zero=1,offload=1", "offload"),
    ],
)
def test_low_precision_masters_need_zero_1_or_2_without_offload_on_every_stage(
    master, stage, needs
):
    args = train_vl.build_parser().parse_args(
        [
            "--stages",
            "2",
            "--stage",
            "0:gpus=4,tp=4,zero=1",
            "--stage",
            stage,
            "--master-dtype",
            master,
            "--grad-dtype",
            "bf16",
        ]
    )
    assert needs in train_vl.optimizer_problem(args)


def test_main_exits_before_loading_anything():
    with pytest.raises(SystemExit):
        train_vl.main([*STAGES, "--master-dtype", "bf16"])


def test_fused_adam_is_off_by_default_and_passed_to_torch_adamw_when_asked():
    assert "fused" not in train_vl.ds_config(parse())["optimizer"]["params"]
    params = train_vl.ds_config(parse("--fused-adam"))["optimizer"]["params"]
    assert params["fused"] is True and params["torch_adam"] is True
    assert train_vl.optimizer_problem(parse("--fused-adam")) is None
    assert train_vl.optimizer_problem(parse("--fused-adam", "--grad-dtype", "bf16")) is None


def test_fused_adam_refused_under_offload():
    args = train_vl.build_parser().parse_args(
        [
            "--stages",
            "2",
            "--stage",
            "0:gpus=4,tp=4,zero=1",
            "--stage",
            "1:gpus=4,tp=4,zero=1,offload=1",
            "--fused-adam",
        ]
    )
    assert "stage 1 offloads" in train_vl.optimizer_problem(args)


def test_stage_key_recompute_blocks():
    (stage,) = parse().stage[:1]
    assert stage.recompute_blocks is None
    args = train_vl.build_parser().parse_args(
        ["--stages", "1", "--stage", "0:gpus=8,tp=2,recompute_blocks=24"]
    )
    assert args.stage[0].recompute_blocks == 24 and not args.stage[0].recompute
