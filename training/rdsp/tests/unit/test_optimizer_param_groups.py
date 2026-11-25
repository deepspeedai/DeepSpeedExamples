# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Optimizer parameter groups for DeepSpeed's ZeRO step: consecutive, at most
max_numel elements each (a larger parameter alone), every parameter exactly
once, and built only when DeepSpeed lists them (after its tensor-parallel
split replaced modules)."""

import torch

from ray_deepspeed_pipeline.config import OPTIMIZER_GROUP_NUMEL_KEY
from ray_deepspeed_pipeline.deepspeed_adapter import optimizer_param_groups


def model(*sizes):
    return torch.nn.ParameterList(torch.nn.Parameter(torch.zeros(n)) for n in sizes)


def numels(groups):
    return [[p.numel() for p in g["params"]] for g in groups]


def test_consecutive_groups_of_at_most_max_numel():
    assert numels(optimizer_param_groups(model(3, 3, 3, 3, 3), 6)) == [[3, 3], [3, 3], [3]]


def test_a_parameter_larger_than_max_numel_is_a_group_of_its_own():
    assert numels(optimizer_param_groups(model(2, 10, 2, 2), 4)) == [[2], [10], [2, 2]]


def test_every_parameter_once_in_module_order():
    m = model(5, 1, 7, 2, 2, 9)
    flat = [p for g in optimizer_param_groups(m, 8) for p in g["params"]]
    assert flat == list(m.parameters())


def test_groups_are_built_when_listed_not_when_asked_for():
    m = torch.nn.Module()
    m.a = torch.nn.Linear(4, 4, bias=False)
    groups = optimizer_param_groups(m, 100)
    m.a = torch.nn.Linear(4, 2, bias=False)  # as a tensor-parallel split replaces a module
    (only,) = list(groups)
    assert only["params"] == [m.a.weight]


def test_train_vl_passes_the_size_under_rdsp_s_own_key():
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    import train_vl

    args = train_vl.build_parser().parse_args(["--optimizer-group-numel", "250000000"])
    assert train_vl.ds_config(args)[OPTIMIZER_GROUP_NUMEL_KEY] == 250_000_000
    assert OPTIMIZER_GROUP_NUMEL_KEY not in train_vl.ds_config(
        train_vl.build_parser().parse_args([])
    )
