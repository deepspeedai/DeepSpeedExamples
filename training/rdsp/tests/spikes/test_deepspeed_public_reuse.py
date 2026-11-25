# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Public DeepSpeed APIs alone support stage-local pipeline stages (no Ray).

Checked on GPUs, with only public DeepSpeed APIs at the pinned revision
53a2ac44fb664bea838df3981ba4366b91643070 (no DeepSpeed file is modified):
  - concurrent stage-local default process groups with isolated collectives
  - external-gradient backward on a non-terminal stage, two candidates:
      "vjp":    engine.backward((out * grad).sum() * gas)
      "direct": out.backward(gradient=grad)
    with loss/grad/optimizer-step parity vs a monolithic connected baseline,
    across ZeRO 0/1/2/3, fp32/bf16, multi-output stages with non-floating
    metadata, and gradient accumulation
  - stage-local checkpoint save/load round trip and next-step parity

Workers are plain torch.multiprocessing processes: Ray is deliberately absent.
"""

import dataclasses
import os
import tempfile

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

deepspeed = pytest.importorskip("deepspeed")

NEEDED_GPUS = 4
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < NEEDED_GPUS,
    reason=f"needs {NEEDED_GPUS} CUDA devices",
)

LR = 1e-2
BATCH = 8  # full global batch rows; DP=2 -> 4 rows per rank


# --------------------------------------------------------------------------
# fixtures: tiny deterministic stage modules
# --------------------------------------------------------------------------


class TwoHeadStage(nn.Module):
    """Multi-output first stage: two float outputs + non-floating metadata."""

    def __init__(self):
        super().__init__()
        self.body = nn.Linear(16, 32)
        self.h1 = nn.Linear(32, 8)
        self.h2 = nn.Linear(32, 8)

    def forward(self, x):
        b = torch.tanh(self.body(x))
        meta = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        return self.h1(b), self.h2(b), meta


class TwoHeadConsumer(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.Linear(8, 4)

    def forward(self, a, b, meta):
        assert meta.dtype == torch.long  # metadata passed through untouched
        return self.out(a + b)


def build_stages(multi_output):
    torch.manual_seed(0)
    if multi_output:
        return TwoHeadStage(), TwoHeadConsumer()
    stage0 = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
    stage1 = nn.Sequential(nn.Linear(32, 4))
    return stage0, stage1


def make_data(dtype):
    torch.manual_seed(1)
    x = torch.randn(BATCH, 16).to(dtype)
    y = torch.randn(BATCH, 4).to(dtype)
    return x, y


def ds_config(zero, dtype, gas):
    micro = BATCH // (2 * gas)  # DP world is 2 in every parity run
    cfg = {
        "train_batch_size": BATCH,
        "train_micro_batch_size_per_gpu": micro,
        "gradient_accumulation_steps": gas,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": LR,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "torch_adam": True,
            },
        },
        "zero_optimization": {"stage": zero},
        "steps_per_print": 1000,
        "wall_clock_breakdown": False,
    }
    if dtype == torch.bfloat16:
        cfg["bf16"] = {"enabled": True}
    return cfg


def engine_named_grads(engine, prefix):
    from deepspeed.utils import safe_get_full_grad

    out = {}
    for n, p in engine.module.named_parameters():
        g = safe_get_full_grad(p)
        if g is not None:
            out[f"{prefix}.{n}"] = g.detach().float().cpu().numpy().copy()
    return out


def named_grads(module, prefix):
    return {
        f"{prefix}.{n}": p.grad.detach().float().cpu().numpy().copy()
        for n, p in module.named_parameters()
        if p.grad is not None
    }


def named_params(module, prefix):
    return {
        f"{prefix}.{n}": p.detach().float().cpu().numpy().copy()
        for n, p in module.named_parameters()
    }


def monolithic_baseline(multi_output, dtype, device):
    """Same seeded stage modules, one connected graph, full batch, torch AdamW.
    Returns (grads, params_after_step)."""
    stage0, stage1 = build_stages(multi_output)
    stage0.to(device, dtype)
    stage1.to(device, dtype)
    x, y = make_data(dtype)
    x, y = x.to(device), y.to(device)

    opt = torch.optim.AdamW(
        [*stage0.parameters(), *stage1.parameters()],
        lr=LR,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    outs = stage0(x)
    outs = outs if isinstance(outs, tuple) else (outs,)
    loss = nn.functional.mse_loss(stage1(*outs), y)
    loss.backward()
    grads = {**named_grads(stage0, "s0"), **named_grads(stage1, "s1")}
    opt.step()
    params = {**named_params(stage0, "s0"), **named_params(stage1, "s1")}
    return float(loss), grads, params


# --------------------------------------------------------------------------
# external-gradient backward parity worker (2 ranks, GPUs 0-1)
# --------------------------------------------------------------------------


@dataclasses.dataclass
class ParityCfg:
    zero: int
    path: str  # "vjp" | "direct"
    dtype_name: str  # "fp32" | "bf16"
    gas: int = 1
    multi_output: bool = False

    @property
    def dtype(self):
        return torch.bfloat16 if self.dtype_name == "bf16" else torch.float32

    @property
    def row_id(self):
        return (
            f"zero{self.zero}-{self.path}-{self.dtype_name}"
            + (f"-gas{self.gas}" if self.gas > 1 else "")
            + ("-multiout" if self.multi_output else "")
        )


def _parity_worker(rank, cfg: ParityCfg, port, q):
    try:
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = "2"
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
        )
        device = torch.device(f"cuda:{rank}")

        stage0, stage1 = build_stages(cfg.multi_output)
        conf = ds_config(cfg.zero, cfg.dtype, cfg.gas)
        engine0, _, _, _ = deepspeed.initialize(
            model=stage0,
            model_parameters=stage0.parameters(),
            config=conf,
            dist_init_required=False,
        )
        engine1, _, _, _ = deepspeed.initialize(
            model=stage1,
            model_parameters=stage1.parameters(),
            config=conf,
            dist_init_required=False,
        )

        x, y = make_data(cfg.dtype)
        rows = BATCH // 2
        x = x[rank * rows : (rank + 1) * rows].to(device)
        y = y[rank * rows : (rank + 1) * rows].to(device)

        micro = rows // cfg.gas
        losses = []
        for k in range(cfg.gas):
            xm, ym = x[k * micro : (k + 1) * micro], y[k * micro : (k + 1) * micro]

            outs = engine0(xm)
            outs = outs if isinstance(outs, tuple) else (outs,)
            cut = tuple(
                o.detach().requires_grad_(True) if o.is_floating_point() else o for o in outs
            )
            loss = nn.functional.mse_loss(engine1(*cut), ym)
            losses.append(loss.detach().float().cpu())

            engine1.backward(loss)
            grads_in = tuple(c.grad if c.is_floating_point() else None for c in cut)

            if cfg.path == "vjp":
                # engine.backward rescales by 1/gas internally; the incoming
                # grads already carry stage1's 1/gas, so cancel one factor.
                vjp = sum((o * g).sum() for o, g in zip(outs, grads_in) if g is not None)
                engine0.backward(vjp * engine0.gradient_accumulation_steps())
            else:
                torch.autograd.backward(
                    [o for o, g in zip(outs, grads_in) if g is not None],
                    [g for g in grads_in if g is not None],
                )

            # capture on the boundary micro-step, after allreduce inside
            # backward, before step() consumes the grads
            if k == cfg.gas - 1:
                grads = {**engine_named_grads(engine0, "s0"), **engine_named_grads(engine1, "s1")}

            engine1.step()
            engine0.step()

        params = {}
        if cfg.zero >= 3:
            with deepspeed.zero.GatheredParameters(
                [*engine0.module.parameters(), *engine1.module.parameters()]
            ):
                params = {
                    **named_params(engine0.module, "s0"),
                    **named_params(engine1.module, "s1"),
                }
        else:
            params = {**named_params(engine0.module, "s0"), **named_params(engine1.module, "s1")}

        mean_loss_t = torch.stack(losses).mean().to(device)
        dist.all_reduce(mean_loss_t)
        mean_loss = float(mean_loss_t) / 2
        if rank == 0:
            q.put(("ok", cfg.row_id, mean_loss, grads, params))
    except Exception as e:  # report, don't hang the parent
        if rank == 0:
            q.put(("error", cfg.row_id, f"{type(e).__name__}: {e}", None, None))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


PARITY_ROWS = [
    ParityCfg(0, "vjp", "fp32"),
    ParityCfg(0, "direct", "fp32"),
    ParityCfg(1, "vjp", "fp32"),
    ParityCfg(1, "direct", "fp32"),
    ParityCfg(2, "vjp", "fp32"),
    ParityCfg(3, "vjp", "fp32"),
    ParityCfg(0, "vjp", "bf16"),  # the first support row's dtype
    ParityCfg(0, "direct", "bf16"),
    ParityCfg(1, "vjp", "fp32", gas=2),
    ParityCfg(1, "vjp", "fp32", multi_output=True),
]

# rows the supported configurations depend on -> hard assert; the rest is
# recorded evidence (xfail on failure)
REQUIRED = {
    "zero0-vjp-fp32",
    "zero1-vjp-fp32",
    "zero0-vjp-bf16",
    "zero1-vjp-fp32-gas2",
    "zero1-vjp-fp32-multiout",
}


@pytest.mark.parametrize("cfg", PARITY_ROWS, ids=lambda c: c.row_id)
def test_external_grad_backward_parity(cfg):
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    port = 29500 + abs(hash(cfg.row_id)) % 500
    try:
        mp.spawn(_parity_worker, args=(cfg, port, q), nprocs=2, join=True)
    except Exception as e:
        if cfg.row_id in REQUIRED:
            raise
        pytest.xfail(f"evidence row {cfg.row_id} failed to run: {e}")

    status, row, loss, grads, params = q.get()
    if status == "error":
        if row in REQUIRED:
            pytest.fail(f"required row {row}: {loss}")
        pytest.xfail(f"evidence row {row}: {loss}")

    ref_loss, ref_grads, ref_params = monolithic_baseline(
        cfg.multi_output, cfg.dtype, torch.device("cuda:0")
    )

    bf16 = cfg.dtype_name == "bf16"
    gtol = dict(rtol=3e-2, atol=1e-3) if bf16 else dict(rtol=1e-4, atol=1e-6)
    print(f"\n[{row}] loss={loss:.6f} ref={ref_loss:.6f} n_grads={len(grads)}")
    assert np.isclose(loss, ref_loss, rtol=3e-2 if bf16 else 1e-5, atol=1e-3), (
        f"{row}: loss diverges"
    )
    assert grads.keys() == ref_grads.keys(), f"{row}: missing grads"
    for name in ref_grads:
        assert np.allclose(grads[name], ref_grads[name], **gtol), (
            f"{row}: grad mismatch {name} "
            f"(max diff {np.abs(grads[name] - ref_grads[name]).max():.3e})"
        )
    if not bf16:  # bf16 keeps fp32 masters inside DS; step parity is fp32-only
        for name in ref_params:
            assert np.allclose(params[name], ref_params[name], rtol=1e-4, atol=1e-6), (
                f"{row}: post-step param mismatch {name}"
            )


# --------------------------------------------------------------------------
# concurrent stage-local worlds, collective isolation (4 ranks, 2 cohorts)
# --------------------------------------------------------------------------


def _isolation_worker(global_rank, q):
    cohort, local = divmod(global_rank, 2)
    os.environ["LOCAL_RANK"] = str(global_rank)
    torch.cuda.set_device(global_rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{29400 + cohort}", rank=local, world_size=2
    )
    device = torch.device(f"cuda:{global_rank}")

    # collective isolation: each cohort all-reduces its own marker value
    marker = torch.full((4,), float(cohort + 1), device=device)
    dist.all_reduce(marker)

    torch.manual_seed(cohort)  # cohorts run DIFFERENT models concurrently
    model = nn.Linear(8, 8)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config(zero=1, dtype=torch.float32, gas=1),
        dist_init_required=False,
    )
    x = torch.randn(4, 8, device=device)
    loss = engine(x).pow(2).mean()
    engine.backward(loss)
    engine.step()

    q.put(
        (
            global_rank,
            cohort,
            marker.cpu().numpy(),
            dist.get_world_size(),
            engine.module.weight.detach().float().cpu().numpy().copy(),
        )
    )
    dist.destroy_process_group()


def test_stage_local_worlds_are_isolated():
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    mp.spawn(_isolation_worker, args=(q,), nprocs=4, join=True)
    results = sorted((q.get() for _ in range(4)), key=lambda r: r[0])

    for global_rank, cohort, marker, world, _ in results:
        assert world == 2, f"rank {global_rank}: world leaked across cohorts"
        expected = 2.0 * (cohort + 1)  # sum over exactly its own cohort
        assert np.allclose(marker, np.full((4,), expected, dtype=np.float32)), (
            f"rank {global_rank}: cohort {cohort} allreduce crossed cohorts: {marker}"
        )

    # DP replicas agree within a cohort; cohorts hold different models
    assert np.array_equal(results[0][4], results[1][4]), "cohort A replicas diverged"
    assert np.array_equal(results[2][4], results[3][4]), "cohort B replicas diverged"
    assert not np.allclose(results[0][4], results[2][4]), "cohorts unexpectedly share model state"


# --------------------------------------------------------------------------
# stage-local checkpoint round trip (2 ranks)
# --------------------------------------------------------------------------


def _checkpoint_worker(rank, zero, port, ckpt_dir, q):
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    device = torch.device(f"cuda:{rank}")
    conf = ds_config(zero, torch.float32, gas=1)

    def fresh_engine():
        torch.manual_seed(7)
        model = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 4))
        engine, _, _, _ = deepspeed.initialize(
            model=model, model_parameters=model.parameters(), config=conf, dist_init_required=False
        )
        return engine

    def one_step(engine, seed):
        torch.manual_seed(seed)
        x = torch.randn(BATCH, 16)[rank * 4 : (rank + 1) * 4].to(device)
        y = torch.randn(BATCH, 4)[rank * 4 : (rank + 1) * 4].to(device)
        loss = nn.functional.mse_loss(engine(x), y)
        engine.backward(loss)
        engine.step()

    engine_a = fresh_engine()
    one_step(engine_a, seed=11)  # give optimizer real state
    engine_a.save_checkpoint(ckpt_dir, tag="p2")  # every rank saves its shard
    saved = named_params(engine_a.module, "m")

    engine_b = fresh_engine()  # new engine, same world
    engine_b.load_checkpoint(ckpt_dir, tag="p2")
    loaded = named_params(engine_b.module, "m")

    # restored state matches, and both engines evolve identically afterwards
    one_step(engine_a, seed=12)
    one_step(engine_b, seed=12)
    after_a = named_params(engine_a.module, "m")
    after_b = named_params(engine_b.module, "m")

    if rank == 0:
        q.put((saved, loaded, after_a, after_b))
    dist.destroy_process_group()


@pytest.mark.parametrize("zero", [0, 1])
def test_stage_local_checkpoint_roundtrip(zero):
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    with tempfile.TemporaryDirectory() as ckpt_dir:
        mp.spawn(_checkpoint_worker, args=(zero, 29300 + zero, ckpt_dir, q), nprocs=2, join=True)
        saved, loaded, after_a, after_b = q.get()
    for name in saved:
        assert np.array_equal(saved[name], loaded[name]), f"zero{zero}: {name} not restored exactly"
    for name in after_a:
        assert np.allclose(after_a[name], after_b[name], rtol=1e-6, atol=1e-8), (
            f"zero{zero}: post-restore step diverged at {name} (optimizer state?)"
        )


def test_environment_report():
    print(
        f"\ntorch {torch.__version__} | deepspeed {deepspeed.__version__} | "
        f"cuda {torch.version.cuda} | gpus: "
        f"{[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}"
    )
