# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""bf16 gradients under fp32 accumulation: each one is added into the fp32
buffer as soon as autograd produces it and then freed, so no bf16 copy of
the gradients stays allocated (2 bytes per parameter less)."""

import torch

from ray_deepspeed_pipeline.deepspeed_adapter import (
    _free_bf16_grads_after_accumulating,
    _with_immediate_grad_update,
)


def zero1_bf16(**bf16):
    return {
        "bf16": {"enabled": True, **bf16},
        "zero_optimization": {"stage": 1},
        "data_types": {"grad_accum_dtype": "fp32"},
    }


def test_immediate_update_is_on_where_deepspeed_accumulates_bf16_grads_in_fp32():
    assert _with_immediate_grad_update(zero1_bf16())["bf16"]["immediate_grad_update"]
    # an explicit choice stays
    assert not _with_immediate_grad_update(zero1_bf16(immediate_grad_update=False))["bf16"][
        "immediate_grad_update"
    ]
    # other setups are left alone
    for conf in (
        {**zero1_bf16(), "zero_optimization": {"stage": 2}},
        {**zero1_bf16(), "data_types": {"grad_accum_dtype": "bf16"}},
        {"zero_optimization": {"stage": 1}},
    ):
        assert "immediate_grad_update" not in _with_immediate_grad_update(conf).get("bf16", {})


class FakeBF16Optimizer:
    """The two things the adapter relies on: the flag, and the per-parameter
    hook DeepSpeed calls once autograd has accumulated a gradient."""

    immediate_grad_update = True

    def __init__(self, n):
        self.hp = [torch.zeros(n)]

    def accumulate_hp_grads_and_remove_lp(self, lp, group_idx, param_idx):
        self.hp[param_idx].add_(lp.grad.float())


def test_each_bf16_grad_is_freed_once_added_into_fp32():
    w = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    opt = FakeBF16Optimizer(4)
    _free_bf16_grads_after_accumulating(opt)
    for scale in (1.0, 2.0):  # two microbatches
        w.grad = torch.full((4,), scale, dtype=torch.bfloat16)
        opt.accumulate_hp_grads_and_remove_lp(w, 0, 0)
        assert w.grad is None
    assert torch.equal(opt.hp[0], torch.full((4,), 3.0))


def test_other_optimizers_are_left_alone():
    class Plain:
        immediate_grad_update = False

    opt = Plain()
    _free_bf16_grads_after_accumulating(opt)
    assert not hasattr(opt, "accumulate_hp_grads_and_remove_lp")


class FakeDeepSpeedBF16Optimizer(FakeBF16Optimizer):
    """With the buffers DeepSpeed's BF16_Optimizer keeps per parameter, and its own add
    (convert to fp32, then add: two GPU kernels and a temporary)."""

    autoep_folding_tp_group = None

    def __init__(self, n):
        self.fp32_groups_gradients = [[torch.zeros(n)]]
        self.fp32_groups_has_gradients = [[False]]
        self.deepspeed_calls = 0

    def accumulate_hp_grads_and_remove_lp(self, lp, group_idx, param_idx):
        self.deepspeed_calls += 1
        hp = self.fp32_groups_gradients[group_idx][param_idx]
        hp.data.add_(lp.grad.data.to(hp.dtype).view(hp.shape))
        lp._hp_grad = hp
        self.fp32_groups_has_gradients[group_idx][param_idx] = True


def accumulate_microbatches(opt, grads, wrap=True):
    """Each gradient through the optimizer's hook; wrap=False leaves the hook as DeepSpeed's."""
    w = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.bfloat16))
    if wrap:
        _free_bf16_grads_after_accumulating(opt)
    for g in grads:
        w.grad = g.clone()
        opt.accumulate_hp_grads_and_remove_lp(w, 0, 0)
        assert w.grad is None or not wrap
    return w


def test_direct_add_gives_deepspeeds_bits_without_its_conversion():
    torch.manual_seed(0)
    grads = [torch.randn(2, 3).to(torch.bfloat16) * 10**k for k in (-3, 0, 2, -1)]
    reference = FakeDeepSpeedBF16Optimizer(6)
    accumulate_microbatches(reference, grads, wrap=False)
    assert reference.deepspeed_calls == 4

    direct = FakeDeepSpeedBF16Optimizer(6)
    w = accumulate_microbatches(direct, grads)
    assert direct.deepspeed_calls == 0  # the add ran here, as one operation
    assert torch.equal(direct.fp32_groups_gradients[0][0], reference.fp32_groups_gradients[0][0])
    assert direct.fp32_groups_gradients[0][0].dtype == torch.float32
    assert direct.fp32_groups_has_gradients == [[True]]
    assert w._hp_grad is direct.fp32_groups_gradients[0][0]


def test_direct_add_leaves_expert_folding_to_deepspeed():
    # DeepSpeed corrects folded expert gradients before its add: that path stays DeepSpeed's
    opt = FakeDeepSpeedBF16Optimizer(6)
    opt.autoep_folding_tp_group = object()
    accumulate_microbatches(opt, [torch.ones(2, 3, dtype=torch.bfloat16)])
    assert opt.deepspeed_calls == 1
