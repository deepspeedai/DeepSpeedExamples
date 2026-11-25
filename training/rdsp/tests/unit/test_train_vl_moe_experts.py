# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""--moe-experts hf: Liger binds its fused-experts forward onto each routed-experts module
instance (as liger_kernel's _bind_method_to_module does); restore_hf_experts removes that
binding so the class's own forward runs again, and leaves every other patch alone."""

import os
import sys

import pytest
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

import train_vl  # noqa: E402


class Experts(nn.Module):
    def forward(self, x):
        return x + 1


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts, self.shared = Experts(), Experts()


def bind(module, name, fn):  # liger_kernel.transformers.monkey_patch._bind_method_to_module
    module.__dict__[name] = fn.__get__(module, module.__class__)


def test_only_the_routed_experts_get_their_own_forward_back():
    model = nn.Sequential(Block(), Block())
    for block in model:
        bind(block.experts, "forward", lambda self, x: x * 100)
        bind(block.experts, "_get_name", lambda self: "LigerExperts")
        bind(block.shared, "forward", lambda self, x: x * 7)  # a SwiGLU patch: kept
        bind(block.shared, "_get_name", lambda self: "LigerQwen3MoeSwiGLUMLP")
    assert train_vl.restore_hf_experts(model) == 2
    x = torch.ones(2)
    for block in model:
        assert torch.equal(block.experts(x), x + 1)
        assert torch.equal(block.shared(x), x * 7)
    assert train_vl.restore_hf_experts(model) == 0


def test_flag_defaults_to_liger():
    args = train_vl.build_parser().parse_args([])
    assert args.moe_experts == "liger"
    assert train_vl.build_parser().parse_args(["--moe-experts", "hf"]).moe_experts == "hf"


def test_hf_experts_without_liger_is_refused():
    # without --liger nothing installs Liger's expert kernel: "hf" would silently do nothing
    parse = train_vl.build_parser().parse_args
    assert "--liger" in train_vl.experts_problem(parse(["--moe-experts", "hf"]))
    assert train_vl.experts_problem(parse(["--moe-experts", "hf", "--liger"])) is None
    assert train_vl.experts_problem(parse([])) is None


def test_main_exits_on_hf_experts_without_liger(capsys):
    with pytest.raises(SystemExit):
        train_vl.main(["--moe-experts", "hf"])
    assert "--moe-experts hf" in capsys.readouterr().err
