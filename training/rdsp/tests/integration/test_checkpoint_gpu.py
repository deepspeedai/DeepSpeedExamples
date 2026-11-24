# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""The global checkpoint protocol with real DeepSpeed engines and NCCL
stage-local worlds (2 GPUs, one per stage).

Parametrized over ZeRO stages: under ZeRO>=1 the optimizer state and gradient
buffers live inside DeepSpeed's partitioned optimizer, which the CPU stub
cannot show. Test names double as acceptance selectors; keep them stable.
"""

import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

deepspeed = pytest.importorskip("deepspeed")
import ray

import ray_deepspeed_pipeline as rdsp
from ray_deepspeed_pipeline import api
from ray_deepspeed_pipeline.compiler import lower
from ray_deepspeed_pipeline.coordinator import PipelineCoordinator
from ray_deepspeed_pipeline.deepspeed_adapter import _deepspeed_engine_factory
from ray_deepspeed_pipeline.errors import PipelinePoisoned, StepFailed
from ray_deepspeed_pipeline.stage_group import create_stage_clients

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="needs 2 CUDA devices"
)

VOCAB, DIM, SEQ, ROWS, N_MB = 64, 32, 16, 2, 4
FLAG = "/tmp/rdsp_fail_apply"
BWD_FLAG = "/tmp/rdsp_fail_backward"


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(DIM, DIM)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        return x + self.drop(torch.tanh(self.lin(x)))


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        self.blocks = nn.ModuleList(Block() for _ in range(4))
        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


def loss_fn(out, labels):
    return F.cross_entropy(out.float().reshape(-1, VOCAB), labels.reshape(-1))


class _FaultyStep:
    """Real DeepSpeed engine whose step() fails on the terminal stage while
    FLAG exists — a genuine partial apply (stage 0 has already stepped)."""

    def __init__(self, engine, terminal):
        self._engine, self._terminal = engine, terminal

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._engine, name)

    def __call__(self, *a, **k):
        return self._engine(*a, **k)

    def backward(self, loss):
        # stage 0's third backward of a step fails while BWD_FLAG exists:
        # an abandoned step with partial (already-reduced) accumulation
        self._bwd = getattr(self, "_bwd", 0) + 1
        if not self._terminal and os.path.exists(BWD_FLAG) and self._bwd % 4 == 3:
            raise RuntimeError("injected backward failure")
        return self._engine.backward(loss)

    def step(self):
        self._bwd = 0
        if self._terminal and os.path.exists(FLAG):
            raise RuntimeError("injected optimizer-apply failure")
        return self._engine.step()


def faulty_factory(module, conf):
    terminal = len(getattr(module, "post", ())) > 0
    return _FaultyStep(_deepspeed_engine_factory(module, conf), terminal)


def ds_config(zero, optimizer="AdamW"):
    """zero: a ZeRO stage, or "2+offload" for ZeRO-2 with the optimizer in
    host memory (SGD there needs zero_force_ds_cpu_optimizer off)."""
    opt = (
        {"type": "AdamW", "params": {"lr": 1e-2, "torch_adam": True}}
        if optimizer == "AdamW"
        else {"type": "SGD", "params": {"lr": 0.1, "momentum": 0.9}}
    )
    zero_conf = {"stage": zero}
    if zero == "2+offload":
        zero_conf = {"stage": 2, "offload_optimizer": {"device": "cpu"}}
    return {
        "zero_force_ds_cpu_optimizer": False,
        "train_micro_batch_size_per_gpu": ROWS,
        "gradient_accumulation_steps": N_MB,
        "optimizer": opt,
        "zero_allow_untested_optimizer": True,
        "scheduler": {
            "type": "WarmupLR",
            "params": {"warmup_min_lr": 0.0, "warmup_max_lr": 1e-2, "warmup_num_steps": 10},
        },
        "zero_optimization": zero_conf,
        "steps_per_print": 10**6,
    }


@pytest.fixture(scope="module")
def ray_ctx():
    here = os.path.dirname(os.path.abspath(__file__))
    ray.init(
        num_cpus=16,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={"env_vars": {"PYTHONPATH": here}},
    )
    yield
    ray.shutdown()


@pytest.fixture()
def runtime(ray_ctx, monkeypatch):
    made = []

    def factory(*, model, pipeline_config, ds_config, loss_fn, weights=None):
        plan = lower(model, pipeline_config, ds_config)

        def build():
            clients = create_stage_clients(
                model, plan, loss_fn, engine_factory=faulty_factory, use_gpu=True
            )
            made.append(clients)
            return clients

        return PipelineCoordinator(plan, build(), rebuild=build)

    monkeypatch.setattr(api, "_coordinator_factory", factory)
    yield made
    for clients in made:
        for c in clients:
            c.shutdown()


def dataset(n_steps=6):
    g = torch.Generator().manual_seed(3)
    n = n_steps * N_MB * ROWS
    ids = torch.randint(0, VOCAB, (n, SEQ), generator=g)
    return [(ids[i], ids[i].clone()) for i in range(n)]


def make_engine(seed, zero, training_data=None, optimizer="AdamW"):
    torch.manual_seed(seed)
    engine, _, _, _ = rdsp.initialize(
        model=TinyLM(),
        config=ds_config(zero, optimizer),
        training_data=training_data,
        pipeline_config=rdsp.PipelineConfig(stages=2, partition=rdsp.UniformTransformerBlocks()),
        loss_fn=loss_fn,
    )
    return engine


def close(engine):
    """Release an engine's actors (and GPUs) before the next one is built —
    the 2-GPU box holds exactly one pipeline at a time."""
    for w in engine._coordinator._workers:
        w.shutdown()


def fixed(step, data):
    rows = data[step * N_MB * ROWS : (step + 1) * N_MB * ROWS]
    ids = torch.stack([r[0] for r in rows])
    return iter(
        [(ids[k * ROWS : (k + 1) * ROWS], ids[k * ROWS : (k + 1) * ROWS]) for k in range(N_MB)]
    )


def is_ray(obj):
    if type(obj).__module__.split(".")[0] == "ray":
        return True
    if isinstance(obj, dict):
        return any(is_ray(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(is_ray(v) for v in obj)
    return False


@pytest.mark.parametrize("zero", [0, 1])
def test_optimizer_scheduler_rng_data_position_no_objectref(runtime, tmp_path, zero):
    data = dataset()
    a = make_engine(0, zero, training_data=data)
    for _ in range(2):
        a.train_batch()
    assert a.save_checkpoint(str(tmp_path), client_state={"z": zero}) is True
    expected = [float(a.train_batch()) for _ in range(2)]
    close(a)

    b = make_engine(1, zero, training_data=data)
    restored = b.load_checkpoint(str(tmp_path))
    assert not is_ray(restored) and restored[1] == {"z": zero}
    assert b.global_steps == 2
    resumed = [float(b.train_batch()) for _ in range(2)]
    assert resumed == pytest.approx(expected, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("zero", [0, 1])
def test_poisoned_generation_requires_restore(runtime, tmp_path, zero):
    data = dataset()
    ref = make_engine(0, zero)
    ref.train_batch(data_iter=fixed(0, data))
    ref.save_checkpoint(str(tmp_path), "c1")
    clean = float(ref.train_batch(data_iter=fixed(1, data)))
    close(ref)

    engine = make_engine(5, zero)
    engine.load_checkpoint(str(tmp_path), "c1")

    open(FLAG, "w").close()
    try:
        with pytest.raises(PipelinePoisoned):
            engine.train_batch(data_iter=fixed(1, data))
    finally:
        os.remove(FLAG)
    with pytest.raises(PipelinePoisoned):
        engine.train_batch(data_iter=fixed(1, data))
    engine.load_checkpoint(str(tmp_path), "c1")
    retried = float(engine.train_batch(data_iter=fixed(1, data)))
    assert retried == pytest.approx(clean, rel=1e-6, abs=1e-6)


def test_last_committed_whole_pipeline_recovery(runtime, tmp_path):
    made = runtime
    data = dataset()
    engine = make_engine(0, 1)
    engine.train_batch(data_iter=fixed(0, data))
    engine.save_checkpoint(str(tmp_path), "a")
    engine.train_batch(data_iter=fixed(1, data))
    engine.save_checkpoint(str(tmp_path), "b")
    clean3 = float(engine.train_batch(data_iter=fixed(2, data)))

    old = made[-1]
    ray.kill(old[1].actors[0], no_restart=True)
    with pytest.raises(StepFailed):
        engine.train_batch(data_iter=fixed(2, data))
    path, _ = engine.load_checkpoint(str(tmp_path))
    assert path.endswith("b") and engine.global_steps == 2
    assert made[-1] is not old
    recovered = float(engine.train_batch(data_iter=fixed(2, data)))
    assert recovered == pytest.approx(clean3, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("zero", [0, 1, 2, "2+offload"])
def test_abandoned_step_retry_is_exact(runtime, tmp_path, zero):
    """A step that fails mid-backward (after some microbatches were already
    accumulated/reduced — under ZeRO-2 into its running sum) is retried
    without restore; the retry must equal a clean step. SGD so a leaked or
    lost gradient changes the numbers (Adam would hide a uniform scale)."""
    data = dataset()
    ref = make_engine(0, zero, optimizer="SGD")
    ref.train_batch(data_iter=fixed(0, data))
    ref.save_checkpoint(str(tmp_path), "c")
    clean = [float(ref.train_batch(data_iter=fixed(s, data))) for s in (1, 2)]
    close(ref)

    engine = make_engine(9, zero, optimizer="SGD")
    engine.load_checkpoint(str(tmp_path), "c")
    open(BWD_FLAG, "w").close()
    try:
        with pytest.raises(StepFailed):
            engine.train_batch(data_iter=fixed(1, data))
    finally:
        os.remove(BWD_FLAG)
    assert engine.global_steps == 1, "a failed step must not advance"
    retried = [float(engine.train_batch(data_iter=fixed(s, data))) for s in (1, 2)]
    assert retried == pytest.approx(clean, rel=1e-6, abs=1e-6)
