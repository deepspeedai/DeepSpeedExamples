# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""--master-dtype remainder through the stage engine factory: DeepSpeed's
ZeRO-1/2 optimizer steps RemainderAdamW on the model's own bf16 partition,
keeps no master copy, trains as its fp32 master does, and checkpoints the
remainder. CPU, one rank, gloo; needs DeepSpeed."""

import gc
import os
import sys
import weakref

import pytest
import torch

os.environ.setdefault("DS_ACCELERATOR", "cpu")  # not the Mac's MPS
deepspeed = pytest.importorskip("deepspeed")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

import train_vl  # noqa: E402
from ray_deepspeed_pipeline.deepspeed_adapter import _deepspeed_engine_factory  # noqa: E402
from ray_deepspeed_pipeline.remainder_adamw import RemainderAdamW, join_master  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def gloo():
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT="29541", RANK="0", WORLD_SIZE="1", LOCAL_RANK="0"
    )
    deepspeed.init_distributed(dist_backend="gloo")


def engine(master, zero=1, group_numel=0):
    flags = [
        "--stages",
        "1",
        "--stage",
        "0:gpus=1",
        "--zero",
        str(zero),
        "--lr",
        "1e-3",
        "--betas",
        "0.9,0.95",
        "--weight-decay",
        "0.1",
        "--rows",
        "1",
        "--microbatches",
        "1",
        "--grad-dtype",
        "bf16",
        "--master-dtype",
        master,
        "--optimizer-group-numel",
        str(group_numel),
    ]
    args = train_vl.build_parser().parse_args(flags)
    assert train_vl.optimizer_problem(args) is None
    conf = train_vl.ds_config(args)
    conf.update(train_micro_batch_size_per_gpu=1, train_batch_size=1)
    torch.manual_seed(0)
    model = torch.nn.Sequential(*[torch.nn.Linear(64, 64) for _ in range(4)]).to(torch.bfloat16)
    return _deepspeed_engine_factory(model, conf)


def train(e, steps, seed=1):
    g = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(steps):
        x = torch.randn(8, 64, generator=g).to(torch.bfloat16).to(e.device)
        loss = e(x).float().pow(2).mean()
        e.backward(loss)
        e.step()
        losses.append(loss.item())
    return losses


def masters(e):
    """Each group's fp32 master: DeepSpeed's copy, or the weights plus remainder."""
    opt = e.optimizer
    out = []
    for m in opt.single_partition_of_fp32_groups:
        state = opt.optimizer.state.get(m, {})
        out.append(
            join_master(m.detach(), state["remainder"])
            if "remainder" in state
            else m.detach().clone()
        )
    return out


@pytest.mark.parametrize("zero,group_numel", [(1, 0), (2, 0), (1, 5000)])
def test_the_master_is_the_model_s_own_partition(zero, group_numel):
    e = engine("remainder", zero, group_numel)
    opt = e.optimizer
    assert isinstance(e.basic_optimizer, RemainderAdamW)
    assert len(opt.bit16_groups) == (1 if group_numel == 0 else 4)
    for i, m in enumerate(opt.single_partition_of_fp32_groups):
        assert m.dtype == torch.bfloat16
        assert m.data_ptr() == opt.parallel_partitioned_bit16_groups[i][0].data_ptr()
        assert opt.optimizer.param_groups[i]["params"] == [m]


def test_deepspeed_s_master_copy_is_freed():
    import ray_deepspeed_pipeline.deepspeed_adapter as adapter

    copies = []
    real = adapter._master_in_model_weights

    def spy(optimizer):
        copies.extend(weakref.ref(m) for m in optimizer.single_partition_of_fp32_groups)
        real(optimizer)

    adapter._master_in_model_weights = spy
    try:
        e = engine("remainder")
    finally:
        adapter._master_in_model_weights = real
    train(e, 2)
    gc.collect()
    assert copies and all(c() is None for c in copies)


@pytest.mark.parametrize("zero", [1, 2])
def test_trains_as_the_fp32_master_does(zero):
    fp32, rem = engine("fp32", zero), engine("remainder", zero)
    l32, lrem = train(fp32, 20), train(rem, 20)
    assert lrem[0] == l32[0]
    # bf16 weights differ only where a master sits exactly halfway between two
    # bf16 values (rounded away from zero, not to even), so steps stay close
    assert max(abs(a - b) / a for a, b in zip(l32, lrem)) < 1e-2
    for a, b in zip(masters(fp32), masters(rem)):
        assert a.dtype == b.dtype == torch.float32
        assert float((a - b).abs().max()) < 1e-3


def test_checkpoint_round_trip_keeps_the_remainder(tmp_path):
    straight = engine("remainder")
    train(straight, 6)
    train(straight, 6, seed=2)

    first = engine("remainder")
    train(first, 6)
    first.save_checkpoint(str(tmp_path), tag="t")
    resumed = engine("remainder")
    resumed.load_checkpoint(str(tmp_path), tag="t")
    for a, b in zip(masters(first), masters(resumed)):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32))
    train(resumed, 6, seed=2)
    for a, b in zip(masters(straight), masters(resumed)):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32))
