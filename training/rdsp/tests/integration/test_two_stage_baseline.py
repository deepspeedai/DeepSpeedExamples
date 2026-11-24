# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""End-to-end acceptance of the first support row, on 2 GPUs.

Row: BF16, 2 stages, 1 GPU per stage, DP=TP=EP=SP=1, ZeRO-0, identity
boundaries, 1F1B, 4 equal microbatches, Qwen/Qwen3-0.6B loaded untied: the
checkpoint ties embed/lm_head and cross-stage ties are rejected by
partitioning, so the embedding is cloned into the head first.

The path is the full one: PipelineConfig -> ExecutionPlan -> coordinator ->
Ray actors -> StageWorker -> DeepSpeedStageAdapter (real deepspeed.initialize).

Test names double as acceptance selectors; keep them stable.
"""

import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

deepspeed = pytest.importorskip("deepspeed")
transformers = pytest.importorskip("transformers")

import ray

import ray_deepspeed_pipeline as rdsp
from ray_deepspeed_pipeline.compiler import lower
from ray_deepspeed_pipeline.coordinator import PipelineCoordinator
from ray_deepspeed_pipeline.engine import RayPipelineEngine
from ray_deepspeed_pipeline.errors import StepFailed, UnsupportedEngineMethod
from ray_deepspeed_pipeline.partition import build_causal_lm_stage
from ray_deepspeed_pipeline.stage_group import create_stage_clients

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="the first support row needs 2 CUDA devices",
)

MODEL_ID = "Qwen/Qwen3-0.6B"
N_MB, SEQ, LR = 4, 64, 1e-5

DS = {
    "train_batch_size": N_MB,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": N_MB,
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
    "zero_optimization": {"stage": 0},
    "bf16": {"enabled": True},
    "steps_per_print": 1000,
}


def lm_loss(outputs, labels):
    vocab = outputs.shape[-1]
    return F.cross_entropy(outputs[:, :-1, :].float().reshape(-1, vocab), labels[:, 1:].reshape(-1))


def load_untied():
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    # dissolve the embed/lm_head tie: cross-stage tied parameters are rejected
    model.lm_head.weight = torch.nn.Parameter(model.model.embed_tokens.weight.detach().clone())
    model.config.tie_word_embeddings = False
    assert model.lm_head.weight is not model.model.embed_tokens.weight
    return model


def make_batch():
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    text = (
        "Pipeline parallelism splits a model by depth into stages that "
        "process microbatches like an assembly line. "
    ) * 20
    ids = tok(text, return_tensors="pt").input_ids[0, : N_MB * SEQ]
    return ids.reshape(N_MB, SEQ)


def microbatches(batch):
    return [(batch[k : k + 1], batch[k : k + 1].clone()) for k in range(N_MB)]


@pytest.fixture(scope="module")
def first_row():
    ray.init(
        num_cpus=4,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={"env_vars": {"PYTHONPATH": os.path.dirname(os.path.abspath(__file__))}},
    )
    model = load_untied()
    cfg = rdsp.PipelineConfig(stages=2, partition=rdsp.UniformTransformerBlocks())
    plan = lower(model, cfg, DS)
    clients = create_stage_clients(
        model, plan, lm_loss, use_gpu=True, stage_builder=build_causal_lm_stage
    )
    coordinator = PipelineCoordinator(plan, clients)
    engine = RayPipelineEngine(coordinator)
    batch = make_batch()
    yield model, plan, clients, engine, batch
    ray.shutdown()


def test_four_tuple_and_unsupported_local_engine_surface(first_row):
    _, _, _, engine, _ = first_row
    for call in (
        lambda: engine(None),
        lambda: engine.forward(None),
        lambda: engine.backward(None),
        lambda: engine.step(),
    ):
        with pytest.raises(UnsupportedEngineMethod):
            call()
    with pytest.raises(UnsupportedEngineMethod):
        _ = engine.module


def test_parity_loss_and_detached_cpu_loss_no_objectref(first_row):
    model, plan, clients, engine, batch = first_row

    # baseline: the SAME untied bf16 weights, unsplit, one GPU, one graph
    ref = load_untied().to("cuda:0")
    ref_loss = lm_loss(ref(batch.to("cuda:0")).logits, batch.to("cuda:0"))

    # eval first: no update, must match the unsplit forward
    eval_loss = engine.eval_batch(data_iter=iter(microbatches(batch)))
    assert isinstance(eval_loss, torch.Tensor)
    assert eval_loss.shape == () and eval_loss.device.type == "cpu"
    assert eval_loss.requires_grad is False
    assert not str(type(eval_loss)).startswith("<class 'ray"), "no ray types"
    assert float(eval_loss) == pytest.approx(float(ref_loss), rel=2e-2), (
        f"staged eval {float(eval_loss)} vs unsplit {float(ref_loss)}"
    )
    assert engine.global_steps == 0

    # one global optimizer step: same loss (forward precedes the update)
    train_loss = engine.train_batch(data_iter=iter(microbatches(batch)))
    assert float(train_loss) == pytest.approx(float(ref_loss), rel=2e-2)
    assert engine.global_steps == 1


def test_parity_optimizer_update_and_deepspeed_pipeline_behavior(first_row):
    """Repeated steps on one batch must reduce the loss. Exact gradient parity
    of the external-gradient backward is covered by tests/spikes."""
    model, plan, clients, engine, batch = first_row
    losses = [float(engine.train_batch(data_iter=iter(microbatches(batch)))) for _ in range(3)]
    assert losses[-1] < losses[0], f"loss must decrease when overfitting: {losses}"
    assert engine.global_steps >= 4  # 1 from previous test + 3 here


def test_exact_microbatch_consumption(first_row):
    _, _, _, engine, batch = first_row
    steps = engine.global_steps
    # N+1 entries: exactly N consumed, the surplus untouched
    surplus = iter(microbatches(batch) + [("sentinel", "sentinel")])
    engine.train_batch(data_iter=surplus)
    assert next(surplus)[0] == "sentinel"
    # N-1 entries: the whole step fails, nothing advances
    with pytest.raises(StepFailed):
        engine.train_batch(data_iter=iter(microbatches(batch)[: N_MB - 1]))
    assert engine.global_steps == steps + 1


def test_eval_batch_no_update_no_step(first_row):
    _, _, clients, engine, batch = first_row
    steps = engine.global_steps
    before = ray.get(clients[0].actors[0].named_parameters_numpy.remote())
    engine.eval_batch(data_iter=iter(microbatches(batch)))
    after = ray.get(clients[0].actors[0].named_parameters_numpy.remote())
    assert engine.global_steps == steps
    assert all(np.array_equal(before[n], after[n]) for n in before)


def test_environment_report(first_row):
    print(
        f"\ntorch {torch.__version__} | deepspeed {deepspeed.__version__} | "
        f"transformers {transformers.__version__} | "
        f"gpus {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}"
    )
