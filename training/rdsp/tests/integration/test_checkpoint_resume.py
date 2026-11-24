# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Global checkpoint round trip and whole-pipeline recovery on real Ray actors
(CPU, gloo, torch stub engines). test_checkpoint_gpu.py runs the same
protocol with real DeepSpeed engines.

Test names double as acceptance selectors; keep them stable."""

import os

import pytest
import torch
import torch.nn as nn
from test_deepspeed_adapter import N_MB, ROWS, SEQ, VOCAB, loss_fn, stub_engine_factory

import ray_deepspeed_pipeline as rdsp
from ray_deepspeed_pipeline import api
from ray_deepspeed_pipeline.compiler import lower
from ray_deepspeed_pipeline.coordinator import PipelineCoordinator
from ray_deepspeed_pipeline.errors import PipelinePoisoned, StepFailed
from ray_deepspeed_pipeline.stage_group import create_stage_clients

ray = pytest.importorskip("ray")

DS = {
    "train_batch_size": N_MB * ROWS,
    "gradient_accumulation_steps": N_MB,
    "train_micro_batch_size_per_gpu": ROWS,
    "optimizer": {"type": "AdamW", "params": {"lr": 0.05}},
    "scheduler": {"type": "StepLR", "params": {"gamma": 0.5}},
}


class DropBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        return x + self.drop(torch.tanh(self.lin(x)))


class DropToyLM(nn.Module):
    """ToyLM with dropout in every block: stage-local RNG state decides the
    loss, so a resume only matches if RNG was restored too."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, 8)
        self.blocks = nn.ModuleList(DropBlock() for _ in range(6))
        self.norm = nn.LayerNorm(8)
        self.head = nn.Linear(8, VOCAB, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


@pytest.fixture(scope="module")
def ray_ctx(tmp_path_factory):
    here = os.path.dirname(os.path.abspath(__file__))
    unit = os.path.abspath(os.path.join(here, "..", "unit"))
    faults = tmp_path_factory.mktemp("faults")
    flag = str(faults / "fail_apply")
    fail_backward = str(faults / "fail_backward")
    os.environ["RDSP_TEST_FAIL_BACKWARD_PATH"] = fail_backward
    # every stage actor holds one logical CPU while alive; tests keep up to
    # three 2-stage pipelines (plus rebuilds) alive at once
    ray.init(
        num_cpus=16,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={
            "env_vars": {
                "PYTHONPATH": f"{here}:{unit}",
                "RDSP_TEST_FAIL_APPLY": flag,
                "RDSP_TEST_FAIL_BACKWARD": fail_backward,
            }
        },
    )
    yield flag
    ray.shutdown()


@pytest.fixture()
def stub_runtime(ray_ctx, monkeypatch):
    """rdsp.initialize wired to real Ray actors with stub engines, including
    the rebuild factory recovery uses."""
    made = []

    def factory(*, model, pipeline_config, ds_config, loss_fn, weights=None):
        plan = lower(model, pipeline_config, ds_config)

        def build():
            clients = create_stage_clients(
                model, plan, loss_fn, engine_factory=stub_engine_factory, use_gpu=False
            )
            made.append(clients)
            return clients

        return PipelineCoordinator(plan, build(), rebuild=build)

    monkeypatch.setattr(api, "_coordinator_factory", factory)
    yield made
    for clients in made:
        for c in clients:
            c.shutdown()


def dataset(n_steps=6, seed=3):
    g = torch.Generator().manual_seed(seed)
    n = n_steps * N_MB * ROWS
    ids = torch.randint(0, VOCAB, (n, SEQ), generator=g)
    labels = torch.randint(0, VOCAB, (n, SEQ), generator=g)
    return [(ids[i], labels[i]) for i in range(n)]


def make_engine(seed, training_data=None):
    torch.manual_seed(seed)
    engine, _, _, _ = rdsp.initialize(
        model=DropToyLM(),
        config=DS,
        training_data=training_data,
        pipeline_config=rdsp.PipelineConfig(stages=2, partition=rdsp.UniformTransformerBlocks()),
        loss_fn=loss_fn,
    )
    return engine


def fixed_microbatches(step, data):
    rows = data[step * N_MB * ROWS : (step + 1) * N_MB * ROWS]
    ids = torch.stack([r[0] for r in rows])
    labels = torch.stack([r[1] for r in rows])
    return iter(
        [(ids[k * ROWS : (k + 1) * ROWS], labels[k * ROWS : (k + 1) * ROWS]) for k in range(N_MB)]
    )


def contains_ray_type(obj) -> bool:
    if type(obj).__module__.split(".")[0] == "ray":
        return True
    if isinstance(obj, dict):
        return any(contains_ray_type(k) or contains_ray_type(v) for k, v in obj.items())
    if isinstance(obj, (list, tuple, set)):
        return any(contains_ray_type(v) for v in obj)
    return False


def test_optimizer_scheduler_rng_data_position_round_trip(stub_runtime, tmp_path):
    data = dataset()
    a = make_engine(seed=0, training_data=data)
    for _ in range(2):
        a.train_batch()
    assert a.save_checkpoint(str(tmp_path), client_state={"epoch": 0})
    expected = [float(a.train_batch()) for _ in range(2)]

    # a DIFFERENT initialization, fresh loader, fresh actors
    b = make_engine(seed=1, training_data=data)
    path, client_state = b.load_checkpoint(str(tmp_path))
    assert client_state == {"epoch": 0} and path.endswith("global_step2")
    assert b.global_steps == 2
    resumed = [float(b.train_batch()) for _ in range(2)]
    # bitwise: same weights, AdamW moments, StepLR position, dropout RNG
    # stream, and the loader resumes at entry 8 (not 0)
    assert resumed == expected

    # control: the comparison is sensitive to optimizer/scheduler state
    c = make_engine(seed=1, training_data=data)
    c.load_checkpoint(str(tmp_path), load_optimizer_states=False, load_lr_scheduler_states=False)
    assert [float(c.train_batch()) for _ in range(2)] != expected


def test_poisoned_generation_requires_restore(stub_runtime, ray_ctx, tmp_path):
    flag = ray_ctx
    data = dataset()
    engine = make_engine(seed=0)
    engine.train_batch(data_iter=fixed_microbatches(0, data))
    engine.save_checkpoint(str(tmp_path), "c1")

    # reference: a clean pipeline restored from the same commit takes step 1
    # (fresh actor processes do not share an initial RNG state, so the only
    # well-defined reference for a dropout model is one restored from c1)
    ref = make_engine(seed=5)
    ref.load_checkpoint(str(tmp_path), "c1")
    clean_step1 = float(ref.train_batch(data_iter=fixed_microbatches(1, data)))

    open(flag, "w").close()  # terminal stage's optimizer apply now fails
    try:
        with pytest.raises(PipelinePoisoned):
            engine.train_batch(data_iter=fixed_microbatches(1, data))
    finally:
        os.remove(flag)
    # stage 0 applied, stage 1 did not: every call is refused until restore
    with pytest.raises(PipelinePoisoned):
        engine.train_batch(data_iter=fixed_microbatches(1, data))
    with pytest.raises(PipelinePoisoned):
        engine.eval_batch(data_iter=fixed_microbatches(1, data))
    with pytest.raises(PipelinePoisoned):
        engine.save_checkpoint(str(tmp_path), "never")
    assert not os.path.exists(tmp_path / "never")

    engine.load_checkpoint(str(tmp_path), "c1")
    assert engine.global_steps == 1
    # weights, optimizer, RNG consistent again and the failed generation's
    # partial gradients discarded: the retried step matches exactly
    assert float(engine.train_batch(data_iter=fixed_microbatches(1, data))) == clean_step1


def test_last_committed_whole_pipeline_recovery(stub_runtime, tmp_path):
    made = stub_runtime
    data = dataset()
    engine = make_engine(seed=0)
    engine.train_batch(data_iter=fixed_microbatches(0, data))
    engine.save_checkpoint(str(tmp_path), "a")
    engine.train_batch(data_iter=fixed_microbatches(1, data))
    engine.save_checkpoint(str(tmp_path), "b")
    # a later save that fails halfway would leave "b" as the last commit;
    # simulate the leftover of such a save
    os.makedirs(tmp_path / "c" / "stage0")
    clean_step3 = float(engine.train_batch(data_iter=fixed_microbatches(2, data)))

    old_clients = made[-1]
    ray.kill(old_clients[1].actors[0], no_restart=True)
    with pytest.raises(StepFailed) as err:
        engine.train_batch(data_iter=fixed_microbatches(2, data))
    assert not contains_ray_type(err.value.args)

    path, _ = engine.load_checkpoint(str(tmp_path))  # latest -> "b"
    assert path.endswith("b") and engine.global_steps == 2
    assert made[-1] is not old_clients, "dead worker -> whole pipeline rebuilt"
    assert float(engine.train_batch(data_iter=fixed_microbatches(2, data))) == clean_step3


def test_no_objectref_in_save_or_restore_results(stub_runtime, tmp_path):
    engine = make_engine(seed=0, training_data=dataset())
    engine.train_batch()
    saved = engine.save_checkpoint(str(tmp_path), client_state={"x": [1, 2]})
    restored = engine.load_checkpoint(str(tmp_path))
    assert saved is True
    assert isinstance(restored, tuple) and not contains_ray_type(restored)
    assert restored[1] == {"x": [1, 2]}


def test_failed_step_mid_backward_is_retryable(stub_runtime, tmp_path):
    """Stage-local dispatch: stage 0 fails in the middle of a step (after
    partial gradient accumulation, with stage 1 mid-exchange). The step
    fails cleanly; the stage-to-stage links are rebuilt; the retry equals a
    clean step exactly (weights, gradients and RNG untouched by the failure)."""
    flag = os.environ["RDSP_TEST_FAIL_BACKWARD_PATH"]
    data = dataset()
    engine = make_engine(seed=0)
    engine.train_batch(data_iter=fixed_microbatches(0, data))
    engine.save_checkpoint(str(tmp_path), "c")
    clean = float(engine.train_batch(data_iter=fixed_microbatches(1, data)))

    retry = make_engine(seed=4)
    retry.load_checkpoint(str(tmp_path), "c")
    open(flag, "w").close()
    try:
        with pytest.raises(StepFailed) as err:
            retry.train_batch(data_iter=fixed_microbatches(1, data))
    finally:
        os.remove(flag)
    assert not contains_ray_type(err.value.args)
    assert retry.global_steps == 1
    assert float(retry.train_batch(data_iter=fixed_microbatches(1, data))) == clean
