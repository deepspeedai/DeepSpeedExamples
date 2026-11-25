# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Heterogeneous support rows, all on the one runtime.

Each row is declarative data (stage grids, DeepSpeed strategy, model) and runs
through the same PipelineConfig -> ExecutionPlan -> coordinator -> actor-group
path as the first row. Row ids are the -k selectors. Acceptance per row:

    pytest tests/integration/test_heterogeneous_pipeline.py \
        -k "<row-id> and parity and global_microbatch and checkpoint and failure"

The per-row test checks, in order:
  parity            loss trajectory vs a single-process, non-pipelined model
                    with identical initial weights, data, and optimizer
  global_microbatch exactly N entries consumed per step whatever the stages'
                    data-parallel degrees; surplus untouched
  checkpoint        save -> train; fresh pipeline -> load -> identical losses
  failure           a non-zero rank of the widest stage dies -> StepFailed
                    (no Ray types) -> whole-pipeline recovery from the last
                    commit -> training resumes on the committed trajectory

Rows whose stages only use data parallelism also run on CPU (gloo, torch
stub engines averaging gradients like ZeRO-0); every row runs with real
DeepSpeed engines on GPU.
"""

import os
from dataclasses import dataclass, field

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ray = pytest.importorskip("ray")

import ray_deepspeed_pipeline as rdsp
from ray_deepspeed_pipeline import api
from ray_deepspeed_pipeline.compiler import lower
from ray_deepspeed_pipeline.config import StageOverride
from ray_deepspeed_pipeline.coordinator import PipelineCoordinator
from ray_deepspeed_pipeline.errors import StepFailed, ValidationError
from ray_deepspeed_pipeline.hf_stage import build_hf_stage
from ray_deepspeed_pipeline.partition import build_causal_lm_stage
from ray_deepspeed_pipeline.stage_group import create_stage_clients

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT = os.path.abspath(os.path.join(HERE, "..", "unit"))


def _cluster_gpus() -> int:
    """GPUs available to the pipeline: the whole Ray cluster when attached
    to one (multi-node), else this machine's devices."""
    if os.environ.get("RDSP_RAY_ADDRESS"):
        return int(os.environ.get("RDSP_CLUSTER_GPUS", "32"))
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


GPUS = _cluster_gpus()

VOCAB, DIM, SEQ = 128, 64, 32


# --- models -------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(DIM, 2 * DIM)
        self.down = nn.Linear(2 * DIM, DIM)

    def forward(self, x):
        return x + self.down(torch.tanh(self.up(x)))


class TinyLM(nn.Module):
    def __init__(self, n_blocks=8):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        self.blocks = nn.ModuleList(Block() for _ in range(n_blocks))
        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


def tiny_qwen3(moe_layers=()):
    """Randomly initialized small Qwen3 (dense) or Qwen3-MoE whose layers
    outside `moe_layers` are dense — untied, fp32."""
    import transformers

    common = dict(
        vocab_size=VOCAB,
        hidden_size=DIM,
        intermediate_size=2 * DIM,
        # 8 q / 4 kv heads: divisible by tp=4 (AutoTP) and sp=2 (Ulysses)
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=4 * SEQ,
        tie_word_embeddings=False,
        attn_implementation="sdpa",
    )
    if moe_layers:
        cfg = transformers.Qwen3MoeConfig(
            **common,
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=DIM,
            norm_topk_prob=True,
            decoder_sparse_step=1,
            mlp_only_layers=[i for i in range(8) if i not in moe_layers],
        )
        model = transformers.Qwen3MoeForCausalLM(cfg)
    else:
        model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(**common))
    return model.float()


def lm_loss(out, labels):
    out = out.logits if hasattr(out, "logits") else out
    return F.cross_entropy(out[:, :-1].float().reshape(-1, VOCAB), labels[:, 1:].reshape(-1))


# --- support rows --------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    row_id: str
    overrides: tuple  # StageOverride per stage
    model: str = "toy"  # toy | qwen3 | qwen3moe
    zero: int = 0
    cpu_ok: bool = False  # data-parallel-only rows also run on CPU stubs
    rows_per_mb: int = 8
    n_mb: int = 4
    moe_layers: tuple = field(default=())
    builder: str = "causal"  # causal (CausalLMStage) | hf (HFModelStage)

    @property
    def stages(self):
        return len(self.overrides)

    @property
    def gpus(self):
        return sum(o.num_gpus for o in self.overrides)


def S(i, n, **kw):
    return StageOverride(stage=i, num_gpus=n, **kw)


ROWS = [
    # replicate -> shard -> replicate: 1 GPU, 2-way DP, 1 GPU
    Row("static-boundary", (S(0, 1), S(1, 2), S(2, 1)), cpu_ok=True),
    # differing GPU counts and shard-to-shard boundaries (2 -> 4 -> 2)
    Row("resource-mesh", (S(0, 2), S(1, 4), S(2, 2)), cpu_ok=True),
    # differing DP degrees AND ZeRO stages per stage
    Row(
        "dp-zero",
        (S(0, 4, zero_stage=2), S(1, 2, zero_stage=1), S(2, 2, zero_stage=0)),
        cpu_ok=True,
    ),
    # optimizer state in host memory on the ZeRO stages, mixed with a plain one
    Row(
        "optimizer-offload",
        (
            S(0, 2, zero_stage=2, offload_optimizer=True),
            S(1, 2, zero_stage=1, offload_optimizer=True),
            S(2, 1),
        ),
    ),
    # stage-local AutoTP (tp=2 and tp=2 x dp=2) on a real HF architecture
    Row("autotp", (S(0, 1), S(1, 2, tp=2), S(2, 4, tp=2)), model="qwen3"),
    # stage-local Ulysses sequence parallelism (sp=2, and sp=2 x dp=2)
    Row("sequence-parallel", (S(0, 2, sp=2), S(1, 4, sp=2), S(2, 1)), model="qwen3"),
    # stage-local AutoEP (ep=4) with and without Parallel Folding (tp=2)
    Row(
        "autoep-folding",
        (S(0, 1), S(1, 4, ep=4), S(2, 2, ep=2, tp=2, fold=True)),
        model="qwen3moe",
        moe_layers=(3, 4, 5, 6, 7),
    ),
    # the same three intra-stage layouts on HFModelStage (the model's own forward)
    Row("hf-autotp", (S(0, 1), S(1, 2, tp=2), S(2, 4, tp=2)), model="qwen3", builder="hf"),
    Row(
        "hf-sequence-parallel", (S(0, 2, sp=2), S(1, 4, sp=2), S(2, 1)), model="qwen3", builder="hf"
    ),
    Row(
        "hf-autoep-folding",
        (S(0, 1), S(1, 4, ep=4), S(2, 2, ep=2, tp=2, fold=True)),
        model="qwen3moe",
        moe_layers=(3, 4, 5, 6, 7),
        builder="hf",
    ),
    # the target composed topology, 32 GPUs on 4 nodes (one stage per node);
    # MoE layers only where the EP stage is (layers 4-5 = stage 2)
    Row(
        "four-stage-mixed",
        (
            S(0, 8, sp=2, zero_stage=2),
            S(1, 8, tp=4, zero_stage=1),
            S(2, 8, ep=8, tp=2, fold=True),
            S(3, 8, tp=2),
        ),
        model="qwen3moe",
        moe_layers=(4, 5),
    ),
    # the same four stage KINDS composed at 8-GPU scale (fits one node, or two
    # 4-GPU nodes): composition evidence, not the 32-GPU row itself
    Row(
        "four-stage-mixed-mini",
        (
            S(0, 2, sp=2, zero_stage=2),
            S(1, 2, tp=2, zero_stage=1),
            S(2, 2, ep=2, tp=2, fold=True),
            S(3, 2, tp=2),
        ),
        model="qwen3moe",
        moe_layers=(4, 5),
    ),
]
ROW_IDS = [r.row_id for r in ROWS]


def ds_config(row: Row):
    return {
        "train_batch_size": row.rows_per_mb * row.n_mb,
        "gradient_accumulation_steps": row.n_mb,
        # SGD, not Adam: Adam's update is invariant to a uniform scale of a
        # parameter's gradient, so a stage receiving k x its true gradient
        # would still match — SGD exposes any boundary scaling error.
        # Momentum gives the checkpoint real optimizer state to restore.
        "optimizer": {"type": "SGD", "params": {"lr": 0.2, "momentum": 0.9}},
        "zero_optimization": {"stage": row.zero},
        "zero_allow_untested_optimizer": True,
        "zero_force_ds_cpu_optimizer": False,  # SGD under optimizer offload
        "steps_per_print": 10**6,
    }


def build_model(row: Row, seed: int):
    torch.manual_seed(seed)
    if row.model == "toy":
        return TinyLM()
    return tiny_qwen3(row.moe_layers)


def pipeline_config(row: Row):
    return rdsp.PipelineConfig(
        stages=row.stages, partition=rdsp.UniformTransformerBlocks(), stage_overrides=row.overrides
    )


def dataset(row: Row, n_steps=8):
    """Learnable sequences (arithmetic progressions mod VOCAB with a random
    start and stride): the loss falls fast, so every step visibly moves it."""
    g = torch.Generator().manual_seed(11)
    n = n_steps * row.n_mb * row.rows_per_mb
    start = torch.randint(0, VOCAB, (n, 1), generator=g)
    stride = torch.randint(1, 4, (n, 1), generator=g)
    ids = (start + stride * torch.arange(SEQ)) % VOCAB
    return [(ids[i], ids[i].clone()) for i in range(n)]


def step_entries(row: Row, data, step):
    """The N global-microbatch entries of one step, as the loader yields them."""
    per = row.rows_per_mb
    base = step * row.n_mb * per
    out = []
    for k in range(row.n_mb):
        rows = data[base + k * per : base + (k + 1) * per]
        ids = torch.stack([r[0] for r in rows])
        out.append((ids, ids.clone()))
    return out


# --- runtime wiring --------------------------------------------------------------


@pytest.fixture(scope="module")
def ray_ctx():
    runtime_env = {"env_vars": {"PYTHONPATH": f"{HERE}:{UNIT}"}}
    address = os.environ.get("RDSP_RAY_ADDRESS")
    if address:  # multi-node: attach to the running cluster
        ray.init(address=address, log_to_driver=False, runtime_env=runtime_env)
    else:
        ray.init(num_cpus=64, include_dashboard=False, log_to_driver=False, runtime_env=runtime_env)
    yield
    ray.shutdown()


def requirements(row: Row):
    if GPUS >= row.gpus:
        pytest.importorskip("deepspeed")
        if row.model != "toy":
            pytest.importorskip("transformers")
        return True  # GPU with real DeepSpeed
    if row.cpu_ok and GPUS == 0:
        return False  # CPU stubs
    pytest.skip(f"{row.row_id} needs {row.gpus} GPUs (have {GPUS})")


@pytest.fixture()
def runtime(ray_ctx, monkeypatch):
    made = []

    def install(row: Row, use_gpu: bool):
        from test_deepspeed_adapter import stub_engine_factory

        factory_kw = {"use_gpu": use_gpu}
        if not use_gpu:
            factory_kw["engine_factory"] = stub_engine_factory
        if row.model != "toy":
            factory_kw["stage_builder"] = (
                build_hf_stage if row.builder == "hf" else build_causal_lm_stage
            )

        def factory(*, model, pipeline_config, ds_config, loss_fn, weights=None):
            plan = lower(model, pipeline_config, ds_config)

            def build():
                clients = create_stage_clients(model, plan, loss_fn, **factory_kw)
                made.append(clients)
                return clients

            return PipelineCoordinator(plan, build(), rebuild=build)

        monkeypatch.setattr(api, "_coordinator_factory", factory)
        return made

    yield install
    for clients in made:
        for c in clients:
            c.shutdown()


def make_engine(row, seed, data=None):
    engine, _, _, _ = rdsp.initialize(
        model=build_model(row, seed),
        config=ds_config(row),
        training_data=data,
        pipeline_config=pipeline_config(row),
        loss_fn=lm_loss,
    )
    return engine


def reference_losses(row, data, steps, use_gpu):
    """Single-process, non-pipelined baseline: same init, same global batch
    per step (the mean over its N microbatches), same SGD as ds_config."""
    model = build_model(row, seed=0)
    device = "cuda:0" if use_gpu else "cpu"
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9)
    losses = []
    for s in range(steps):
        opt.zero_grad()
        total = 0.0
        for ids, labels in step_entries(row, data, s):
            loss = lm_loss(model(ids.to(device)), labels.to(device)) / row.n_mb
            loss.backward()
            total += float(loss.detach())
        opt.step()
        losses.append(total)
    return losses


def close(engine):
    """Release an engine's actors and placement groups before the next one is
    built: a GPU box holds one pipeline of a row at a time."""
    for w in engine._coordinator._workers:
        w.shutdown()


def is_ray(obj):
    if type(obj).__module__.split(".")[0] == "ray":
        return True
    if isinstance(obj, (list, tuple)):
        return any(is_ray(v) for v in obj)
    return False


# --- the per-row acceptance test -----------------------------------------------


@pytest.mark.parametrize("row", ROWS, ids=ROW_IDS)
def test_row_parity_global_microbatch_checkpoint_failure(row, runtime, tmp_path):
    use_gpu = requirements(row)
    made = runtime(row, use_gpu)
    data = dataset(row)
    steps = 3
    # fp32 on both sides: only reduction order differs. Tight enough that a
    # stage applying a partial gradient (e.g. one microbatch of N) fails.
    tol = dict(rel=1e-4, abs=1e-5) if use_gpu else dict(rel=1e-5, abs=1e-6)

    # parity + global microbatch (engine-owned loader: one entry = one
    # global microbatch, split over each stage's own dp ranks)
    ref = reference_losses(row, data, steps + 2, use_gpu)
    # sensitivity precondition: each optimizer step moves the loss by far
    # more than the tolerance, so a wrong update cannot hide inside it
    moves = [abs(a - b) / abs(b) for a, b in zip(ref[1:], ref)]
    assert min(moves) > 20 * tol["rel"], f"test not sensitive enough: {moves}"
    engine = make_engine(row, seed=0, data=data)
    pipe = [float(engine.train_batch()) for _ in range(steps)]
    assert pipe == pytest.approx(ref[:steps], **tol), f"{row.row_id} parity"
    assert engine._owned_iter.consumed == steps * row.n_mb, "exact consumption"

    # checkpoint round trip across differing per-stage shard sets
    assert engine.save_checkpoint(str(tmp_path), "c3")
    close(engine)

    # global microbatch with a caller-owned iterator: exactly N entries taken
    probe = make_engine(row, seed=0)
    surplus = iter(step_entries(row, data, steps) + [("sentinel", "sentinel")])
    probe.train_batch(data_iter=surplus)
    assert next(surplus)[0] == "sentinel", "surplus entry must stay unconsumed"
    close(probe)

    # a fresh pipeline (different init) restored from c3 continues the
    # reference trajectory; its loader resumes at entry steps*n_mb
    fresh = make_engine(row, seed=7, data=data)
    fresh.load_checkpoint(str(tmp_path), "c3")
    cont = [float(fresh.train_batch()) for _ in range(2)]
    assert cont == pytest.approx(ref[steps : steps + 2], **tol), "resume continues trajectory"
    assert fresh.save_checkpoint(str(tmp_path), "c5")

    # failure: kill a non-zero rank of the widest stage, recover whole pipeline
    widest = max(range(row.stages), key=lambda i: row.overrides[i].num_gpus)
    victim_rank = row.overrides[widest].num_gpus - 1
    ray.kill(made[-1][widest].actors[victim_rank], no_restart=True)
    with pytest.raises(StepFailed) as err:
        fresh.train_batch()
    assert not is_ray(err.value.args)
    before = made[-1]
    path, _ = fresh.load_checkpoint(str(tmp_path), "c3")
    assert made[-1] is not before and fresh.global_steps == steps
    assert [float(fresh.train_batch()) for _ in range(2)] == pytest.approx(cont, **tol)
    # and the latest commit (c5, written after resume) is what `latest` names
    fresh.load_checkpoint(str(tmp_path))
    assert fresh.global_steps == steps + 2


# --- gradient parity, parameter by parameter ------------------------------------


def _reference_name(stage_name: str, block_start: int) -> str | None:
    """Stage parameter name -> the unsplit HF model's name. HFModelStage keeps
    the model's names under `model.`; CausalLMStage renames them."""
    if stage_name.startswith("model."):
        return stage_name[len("model.") :]
    for prefix, target in (
        ("emb.", "model.embed_tokens."),
        ("norm.", "model.norm."),
        ("head.", "lm_head."),
    ):
        if stage_name.startswith(prefix):
            return target + stage_name[len(prefix) :]
    if stage_name.startswith("layers."):
        _, i, rest = stage_name.split(".", 2)
        return f"model.layers.{block_start + int(i)}.{rest}"
    return None


def _rank0_slice(ref, shape):
    """Rank 0's shard of a TP-split weight (column split: leading rows;
    row split: leading columns)."""
    if tuple(ref.shape) == tuple(shape):
        return ref
    if ref.ndim == 2 and ref.shape[1] == shape[1] and ref.shape[0] % shape[0] == 0:
        return ref[: shape[0]]
    if ref.ndim == 2 and ref.shape[0] == shape[0] and ref.shape[1] % shape[1] == 0:
        return ref[:, : shape[1]]
    return None


@pytest.mark.parametrize("row", [r for r in ROWS if r.model != "toy"], ids=lambda r: r.row_id)
def test_row_gradient_parity_per_parameter(row, runtime):
    """One SGD step (no momentum history yet: update = -lr * gradient). For
    every parameter rank 0 of each stage holds, the update must match the
    unsplit model's — TP shards against the matching slice. Parameters AutoEP
    re-creates under new names (experts, router) are covered by loss parity."""
    import numpy as np

    use_gpu = requirements(row)
    runtime(row, use_gpu)
    data = dataset(row)
    ref_model = build_model(row, seed=0).to("cuda:0")
    before = {n: p.detach().cpu().clone() for n, p in ref_model.named_parameters()}
    opt = torch.optim.SGD(ref_model.parameters(), lr=0.2, momentum=0.9)
    for ids, labels in step_entries(row, data, 0):
        (lm_loss(ref_model(ids.cuda()), labels.cuda()) / row.n_mb).backward()
    opt.step()
    ref_delta = {n: p.detach().cpu() - before[n] for n, p in ref_model.named_parameters()}

    engine = make_engine(row, seed=0, data=data)
    engine.train_batch()
    plan_stages = engine._coordinator._plan.stages
    report, checked = [], 0
    for spec, client in zip(plan_stages, engine._coordinator._workers):
        after = ray.get(client.actors[0].named_parameters_numpy.remote())
        for name, value in after.items():
            ref_name = _reference_name(name, spec.block_start)
            if ref_name not in before:
                continue
            start = _rank0_slice(before[ref_name], value.shape)
            want = _rank0_slice(ref_delta[ref_name], value.shape)
            if start is None:
                continue
            got = torch.from_numpy(np.asarray(value)) - start
            err = float((got - want).abs().max() / want.abs().max().clamp_min(1e-12))
            report.append((err, f"stage{spec.index}:{name}"))
            checked += 1
    close(engine)
    report.sort(reverse=True)
    print(f"\n{row.row_id}: {checked} parameters checked; worst: {report[:5]}")
    assert checked > 0
    assert report[0][0] < 1e-3, f"update mismatch: {report[:8]}"


# --- compile-time / placement failure behavior (no GPUs needed) -------------------


def test_failure_batch_divisibility_rejected_before_actors():
    row = ROWS[1]
    ok = dict(ds_config(row), train_batch_size=4 * row.n_mb)  # 4 rows over dp=4
    lower(TinyLM(), pipeline_config(row), ok)
    bad = dict(ds_config(row), train_batch_size=6 * row.n_mb)  # 6 rows over dp=4
    with pytest.raises(ValidationError, match="split evenly"):
        lower(TinyLM(), pipeline_config(row), bad)


@pytest.mark.parametrize(
    "override,match",
    [
        (S(1, 3, tp=2), "divisible by tp"),
        (S(1, 4, ep=3), "ep 3 does not divide"),
        (S(1, 2, fold=True), "fold=True requires ep"),
    ],
)
def test_failure_degree_rejected_before_actors(override, match):
    cfg = rdsp.PipelineConfig(
        stages=3, partition=rdsp.UniformTransformerBlocks(), stage_overrides=(override,)
    )
    with pytest.raises(ValidationError, match=match):
        lower(TinyLM(), cfg, {"train_batch_size": 48, "gradient_accumulation_steps": 4})


def test_failure_sequence_parallel_on_terminal_stage_rejected():
    cfg = rdsp.PipelineConfig(
        stages=2, partition=rdsp.UniformTransformerBlocks(), stage_overrides=(S(1, 2, sp=2),)
    )
    with pytest.raises(ValidationError, match="terminal"):
        lower(TinyLM(), cfg, {"train_batch_size": 8, "gradient_accumulation_steps": 4})


def test_failure_mesh_mismatch_and_duplicate_rank():
    from ray_deepspeed_pipeline.boundary import Grid
    from ray_deepspeed_pipeline.stage_group import check_mesh

    g = Grid(dp=2, tp=2)
    good = [
        {"dp_rank": d, "dp_world": 2, "tp_rank": t, "tp_world": 2}
        for d in range(2)
        for t in range(2)
    ]
    check_mesh(1, g, good)
    swapped = [dict(m) for m in good]
    swapped[1]["tp_rank"], swapped[1]["dp_rank"] = 0, 1  # DeepSpeed put rank 1 elsewhere
    with pytest.raises(ValidationError, match="mesh mismatch"):
        check_mesh(1, g, swapped)
    with pytest.raises(ValidationError, match="bootstrapped"):
        check_mesh(1, g, good[:3])  # a rank missing


@pytest.mark.skipif(GPUS == 0, reason="placement is only constrained on GPU")
def test_failure_resource_loss_unplaceable_stage(ray_ctx, monkeypatch):
    import ray_deepspeed_pipeline.stage_group as stage_group

    monkeypatch.setattr(stage_group, "_PLACEMENT_TIMEOUT_S", 20.0)
    row = Row("too-big", (S(0, 1), S(1, GPUS + 1)))
    plan = lower(
        TinyLM(), pipeline_config(row), dict(ds_config(row), train_batch_size=(GPUS + 1) * row.n_mb)
    )
    with pytest.raises(ValidationError, match="cannot place"):
        create_stage_clients(TinyLM(), plan, lm_loss, use_gpu=True)
