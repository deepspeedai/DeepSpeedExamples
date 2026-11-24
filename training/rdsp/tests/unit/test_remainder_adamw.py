# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""RemainderAdamW: an fp32 master kept as the bf16 weight plus a 16-bit
remainder is exact, and AdamW on it follows torch.optim.AdamW on a separate
fp32 master bit for bit, whatever the chunking."""

import pytest
import torch

from ray_deepspeed_pipeline.remainder_adamw import RemainderAdamW, join_master, split_master

EDGES = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    1e-40,
    -1e-40,
    3.0e38,
    -3.0e38,
    1e-8,
    65504.0,
    float("inf"),
    float("-inf"),
]


def masters():
    g = torch.Generator().manual_seed(0)
    x = torch.cat(
        [
            torch.randn(10_000, generator=g) * 10 ** torch.randint(-6, 6, (10_000,), generator=g),
            torch.tensor(EDGES),
        ]
    )
    # every low half-word pattern around the rounding point, both signs
    low = torch.tensor([0x0000, 0x0001, 0x7FFF, 0x8000, 0x8001, 0xFFFF], dtype=torch.int32)
    for high in (0x3F80, 0xBF80, 0x0000, 0x8000, 0x7F7F, 0xFF7F):
        x = torch.cat([x, ((high << 16) | low).view(torch.float32)])
    return x


def split(master):
    weight = torch.empty_like(master, dtype=torch.bfloat16)
    remainder = torch.empty_like(master, dtype=torch.int16)
    split_master(master, weight, remainder)
    return weight, remainder


def test_split_then_join_is_exact():
    m = masters()
    assert torch.equal(join_master(*split(m)).view(torch.int32), m.view(torch.int32))


def test_weight_is_the_master_rounded_to_nearest():
    m = masters()
    m = m[torch.isfinite(m) & (m.abs() < 3e38)]  # rounding up past the largest bf16
    weight, _ = split(m)
    rne = m.to(torch.bfloat16)
    tie = (m.view(torch.int32) & 0xFFFF) == 0x8000
    assert torch.equal(weight[~tie].view(torch.int16), rne[~tie].view(torch.int16))
    # a tie rounds away from zero
    assert (weight[tie].float().abs() >= m[tie].abs()).all()


def grads(shape, steps):
    g = torch.Generator().manual_seed(1)
    return [torch.randn(shape, generator=g).to(torch.bfloat16) for _ in range(steps)]


@pytest.mark.parametrize("kernel", [dict(foreach=False), dict(foreach=True), dict(fused=True)])
@pytest.mark.parametrize("chunk", [64, 1000, 1 << 25])
def test_matches_torch_adamw_on_an_fp32_master(kernel, chunk):
    hyper = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    start = torch.randn(2, 1537, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
    ref = torch.nn.Parameter(start.float())
    ref_opt = torch.optim.AdamW([ref], **hyper, **kernel)
    p = torch.nn.Parameter(start.clone())
    opt = RemainderAdamW([p], **hyper, **kernel, chunk_numel=chunk)
    for g in grads(start.shape, 30):
        ref.grad = g.float()
        p.grad = g
        ref_opt.step()
        opt.step()
    master = join_master(p.detach(), opt.state[p]["remainder"])
    assert torch.equal(master.view(torch.int32), ref.detach().view(torch.int32))
    assert torch.equal(opt.state[p]["exp_avg"], ref_opt.state[ref]["exp_avg"])
    assert torch.equal(opt.state[p]["exp_avg_sq"], ref_opt.state[ref]["exp_avg_sq"])
    assert float(opt.state[p]["step"]) == float(ref_opt.state[ref]["step"]) == 30


def test_state_dict_round_trip_keeps_fp32_moments_and_the_remainder():
    hyper = dict(lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
    start = torch.randn(3000, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    gs = grads(start.shape, 20)

    straight = torch.nn.Parameter(start.clone())
    opt = RemainderAdamW([straight], **hyper)
    for g in gs:
        straight.grad = g
        opt.step()

    first = torch.nn.Parameter(start.clone())
    opt1 = RemainderAdamW([first], **hyper)
    for g in gs[:10]:
        first.grad = g
        opt1.step()
    resumed = torch.nn.Parameter(first.detach().clone())
    opt2 = RemainderAdamW([resumed], **hyper)
    opt2.load_state_dict(opt1.state_dict())
    state = opt2.state[resumed]
    assert state["exp_avg"].dtype == torch.float32 and state["remainder"].dtype == torch.int16
    for g in gs[10:]:
        resumed.grad = g
        opt2.step()
    assert torch.equal(resumed.detach().view(torch.int16), straight.detach().view(torch.int16))
    assert torch.equal(opt2.state[resumed]["remainder"], opt.state[straight]["remainder"])


def test_refuses_fp32_parameters():
    p = torch.nn.Parameter(torch.zeros(4))
    p.grad = torch.ones(4)
    with pytest.raises(TypeError, match="bf16"):
        RemainderAdamW([p]).step()
