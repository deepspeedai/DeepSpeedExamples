# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""The full runtime on real Ray actors (CPU, gloo, torch stub engines).

On GPU only the engine changes (real DeepSpeed); the rest of this path is the
same."""

import os

import pytest
import torch
from test_deepspeed_adapter import (
    N_MB,
    ROWS,
    VOCAB,
    loss_fn,
    make_data,
    stub_engine_factory,
)
from test_partition import ToyLM

import ray_deepspeed_pipeline as rdsp
from ray_deepspeed_pipeline.compiler import lower
from ray_deepspeed_pipeline.coordinator import PipelineCoordinator
from ray_deepspeed_pipeline.engine import RayPipelineEngine
from ray_deepspeed_pipeline.schedule import generate_commands
from ray_deepspeed_pipeline.stage_group import create_stage_clients

ray = pytest.importorskip("ray")

DS = {
    "train_batch_size": N_MB * ROWS,
    "gradient_accumulation_steps": N_MB,
    "train_micro_batch_size_per_gpu": ROWS,
    "optimizer": {"type": "AdamW", "params": {"lr": 0.05}},
}


@pytest.fixture(scope="module")
def ray_ctx():
    unit_dir = os.path.join(os.path.dirname(__file__), "..", "unit")
    ray.init(
        num_cpus=3,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={"env_vars": {"PYTHONPATH": os.path.abspath(unit_dir)}},
    )
    yield
    ray.shutdown()


@pytest.fixture()
def runtime(ray_ctx):
    torch.manual_seed(0)
    model = ToyLM()
    cfg = rdsp.PipelineConfig(stages=2, partition=rdsp.UniformTransformerBlocks())
    plan = lower(model, cfg, DS)
    clients = create_stage_clients(
        model, plan, loss_fn, engine_factory=stub_engine_factory, use_gpu=False
    )
    coordinator = PipelineCoordinator(plan, clients)
    engine = RayPipelineEngine(coordinator)
    return model, plan, clients, coordinator, engine


def microbatches(ids, labels):
    return [
        (ids[k * ROWS : (k + 1) * ROWS], labels[k * ROWS : (k + 1) * ROWS]) for k in range(N_MB)
    ]


def test_three_step_loss_trajectory_matches_monolithic(runtime):
    model, plan, clients, coordinator, engine = runtime
    ids, labels = make_data()

    # baseline: same initial weights, one process, one graph, same optimizer
    ref = ToyLM()
    ref.load_state_dict(model.state_dict())
    opt = torch.optim.AdamW(ref.parameters(), lr=0.05)

    staged_losses, ref_losses = [], []
    for _ in range(3):
        staged = engine.train_batch(data_iter=iter(microbatches(ids, labels)))
        staged_losses.append(float(staged))

        ref_loss = loss_fn(ref(ids), labels)
        opt.zero_grad()
        ref_loss.backward()
        opt.step()
        ref_losses.append(float(ref_loss.detach()))

    assert staged_losses == pytest.approx(ref_losses, rel=1e-5), (
        f"staged {staged_losses} vs monolithic {ref_losses}"
    )
    assert staged_losses[2] < staged_losses[0], "training must reduce the loss"
    assert engine.global_steps == 3


def test_only_terminal_actor_holds_loss_fn(runtime):
    _, _, clients, _, _ = runtime
    assert not ray.get(clients[0].actors[0].has_loss_fn.remote())
    assert ray.get(clients[1].actors[0].has_loss_fn.remote())


def test_actors_execute_commands_in_schedule_order(runtime):
    model, plan, clients, coordinator, engine = runtime
    ids, labels = make_data()
    engine.train_batch(data_iter=iter(microbatches(ids, labels)))
    expected = generate_commands(plan, coordinator._generation, engine.global_steps - 1)
    for s, client in enumerate(clients):
        executed = ray.get(client.actors[0].executed_commands.remote())
        this_gen = [c for c in executed if c.startswith(f"g{coordinator._generation}.")]
        assert this_gen == [c.command_id for c in expected[s]], f"stage {s}"


def test_eval_batch_touches_nothing(runtime):
    model, plan, clients, coordinator, engine = runtime
    ids, labels = make_data()
    steps_before = engine.global_steps
    before = ray.get(clients[0].actors[0].named_parameters_numpy.remote())
    loss = engine.eval_batch(data_iter=iter(microbatches(ids, labels)))
    after = ray.get(clients[0].actors[0].named_parameters_numpy.remote())
    assert isinstance(loss, torch.Tensor) and loss.shape == ()
    assert engine.global_steps == steps_before
    assert all((before[n] == after[n]).all() for n in before)


def test_through_public_initialize(ray_ctx, monkeypatch):
    """rdsp.initialize wired to the real Ray runtime (stub engines)."""
    from ray_deepspeed_pipeline import api

    def factory(*, model, pipeline_config, ds_config, loss_fn, weights=None):
        plan = lower(model, pipeline_config, ds_config)
        clients = create_stage_clients(
            model, plan, loss_fn, engine_factory=stub_engine_factory, use_gpu=False
        )
        return PipelineCoordinator(plan, clients)

    monkeypatch.setattr(api, "_coordinator_factory", factory)
    torch.manual_seed(0)
    ids, labels = make_data()
    engine, opt, loader, sched = rdsp.initialize(
        model=ToyLM(),
        config=DS,
        pipeline_config=rdsp.PipelineConfig(stages=2, partition=rdsp.UniformTransformerBlocks()),
        loss_fn=loss_fn,
    )
    assert opt is None and loader is None and sched is None
    loss = engine.train_batch(data_iter=iter(microbatches(ids, labels)))
    assert loss.shape == () and loss.device.type == "cpu"
    assert engine.global_steps == 1


# Recorded from both dispatch paths (stage-local and the former driver-dispatched
# one), which agreed bit for bit at 1 and 3 stages before the driver path was
# removed: three train steps, then one eval.
REFERENCE_LOSSES = [3.2497334480285645, 2.7702560424804688, 2.578868865966797, 2.4435534477233887]


@pytest.mark.parametrize("stages", [1, 3])
def test_losses_match_recorded_reference(ray_ctx, stages):
    """Any pipeline depth reproduces the recorded losses exactly: the
    regression guard that replaced the driver-vs-local equality test."""
    ids, labels = make_data()
    torch.manual_seed(0)
    model = ToyLM()
    plan = lower(
        model, rdsp.PipelineConfig(stages=stages, partition=rdsp.UniformTransformerBlocks()), DS
    )
    clients = create_stage_clients(
        model, plan, loss_fn, engine_factory=stub_engine_factory, use_gpu=False
    )
    engine = RayPipelineEngine(PipelineCoordinator(plan, clients))
    try:
        losses = [
            float(engine.train_batch(data_iter=iter(microbatches(ids, labels)))) for _ in range(3)
        ]
        losses.append(float(engine.eval_batch(data_iter=iter(microbatches(ids, labels)))))
    finally:
        for client in clients:
            client.shutdown()
    assert losses == REFERENCE_LOSSES


def test_public_initialize_builds_hf_causal_lm_stages(ray_ctx, monkeypatch):
    """rdsp.initialize() on a Hugging Face causal LM, through the real default
    factory and no stage_builder: it must pick build_causal_lm_stage by itself
    (the generic builder cannot feed Qwen3's layers their positions). Only the
    DeepSpeed engine is swapped for the CPU stub. The first step's loss (taken
    before the update) must equal the unsplit model's on the same batches."""
    transformers = pytest.importorskip("transformers")
    import torch.nn.functional as F

    from ray_deepspeed_pipeline import stage_group

    real = stage_group.create_stage_clients
    built = []

    def cpu_stub_clients(model, plan, loss_fn, **kw):
        assert "stage_builder" not in kw  # the default path passes none
        clients = real(
            model, plan, loss_fn, engine_factory=stub_engine_factory, use_gpu=False, **kw
        )
        built.append(clients)
        return clients

    monkeypatch.setattr(stage_group, "create_stage_clients", cpu_stub_clients)

    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = transformers.Qwen3ForCausalLM(cfg).float().eval()

    def lm_loss(out, labels):
        return F.cross_entropy(out[:, :-1].reshape(-1, VOCAB), labels[:, 1:].reshape(-1))

    ids, _ = make_data()
    batches = microbatches(ids, ids)
    with torch.no_grad():
        expected = sum(float(lm_loss(model(x).logits, y)) for x, y in batches) / N_MB

    engine, _, _, _ = rdsp.initialize(
        model=model,
        config=DS,
        loss_fn=lm_loss,
        pipeline_config=rdsp.PipelineConfig(stages=2, partition=rdsp.UniformTransformerBlocks()),
    )
    try:
        # the generic builder cannot run Qwen3 layers at all (they need
        # position embeddings), so a matching loss proves the right builder
        loss = float(engine.train_batch(data_iter=iter(batches)))
        assert loss == pytest.approx(expected, rel=1e-5)
        assert engine.global_steps == 1
    finally:
        for c in built[-1]:
            c.shutdown()


def test_too_few_gpus_fails_before_any_actor_starts(ray_ctx):
    """A layout that needs more GPUs than the cluster has raises at once,
    naming both counts, instead of waiting out the placement timeout."""
    model = ToyLM()
    cfg = rdsp.PipelineConfig(stages=2, partition=rdsp.UniformTransformerBlocks())
    plan = lower(model, cfg, DS)
    actors_before = (
        len(ray.util.list_named_actors(all_namespaces=True))
        if hasattr(ray.util, "list_named_actors")
        else None
    )
    have = int(ray.cluster_resources().get("GPU", 0))
    if have >= 2:
        pytest.skip("this cluster has enough GPUs")
    with pytest.raises(rdsp.errors.ValidationError, match=f"needs 2 GPUs.*has {have}"):
        create_stage_clients(model, plan, loss_fn, engine_factory=stub_engine_factory, use_gpu=True)
    if actors_before is not None:
        assert len(ray.util.list_named_actors(all_namespaces=True)) == actors_before


def test_measure_stages_then_shutdown(runtime):
    """measure_stages() times every stage over real actors and reads back a
    MeasuredStages for the plan; shutdown() stops the actors."""
    _, plan, clients, _, engine = runtime
    ids, labels = make_data()
    steps = (entry for _ in range(3) for entry in microbatches(ids, labels))
    measured = engine.measure_stages(steps, steps=3, skip=1)
    assert measured.cuts == (plan.stages[1].block_start,)
    assert len(measured.seconds) == 2 and all(s > 0 for s in measured.seconds)
    assert measured.limit_gib == float("inf")  # CPU: no GPU memory to limit
    # profiling is off again: later steps keep no records
    engine.train_batch(data_iter=iter(microbatches(ids, labels)))
    assert all(not rank["records"] for stage in clients for rank in stage.profile_records())
    engine.shutdown()
    import time

    deadline = time.monotonic() + 30  # ray.kill takes effect asynchronously
    while any(client.alive(timeout=5) for client in clients) and time.monotonic() < deadline:
        time.sleep(0.5)
    assert not any(client.alive(timeout=5) for client in clients)

    # the measured cut builds a new pipeline in the same Ray session, which trains
    model = ToyLM()
    policy = rdsp.BalancedTransformerBlocks(measured=measured)
    cfg = rdsp.PipelineConfig(stages=2, partition=policy)
    replan = lower(model, cfg, DS)
    coordinator = PipelineCoordinator(
        replan,
        create_stage_clients(
            model, replan, loss_fn, engine_factory=stub_engine_factory, use_gpu=False
        ),
    )
    loss = RayPipelineEngine(coordinator).train_batch(data_iter=iter(microbatches(ids, labels)))
    assert torch.isfinite(loss)
    coordinator.shutdown()
