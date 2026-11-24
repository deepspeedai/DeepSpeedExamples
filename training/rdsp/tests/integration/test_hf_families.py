# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Major Hugging Face model families through rdsp.initialize(), on real Ray
CPU actors (torch stub engine): three stages, the middle one data-parallel,
must train exactly like the unsplit model. The stage builder is rdsp's own
choice: the HF builder for all but the few types CausalLMStage is kept for.

Each family has a quirk the builder has to survive: per-layer-type rotary
tables and masks (Gemma3, Qwen3.5), logit soft-capping after the head
(Gemma2, Cohere), tuple-returning blocks with ALiBi (Bloom), positional block
arguments and learned position embeddings (GPT-2), no attention (Mamba)."""

import os

import pytest
import torch
import torch.nn.functional as F
from test_deepspeed_adapter import stub_engine_factory

import ray_deepspeed_pipeline as rdsp
from ray_deepspeed_pipeline.config import StageOverride
from ray_deepspeed_pipeline.hf_stage import build_hf_stage
from ray_deepspeed_pipeline.partition import build_causal_lm_stage, select_stage_builder

ray = pytest.importorskip("ray")
transformers = pytest.importorskip("transformers")

VOCAB, SEQ, ROWS, N_MB = 128, 10, 2, 2
DS = {
    "train_batch_size": N_MB * ROWS,
    "gradient_accumulation_steps": N_MB,
    "optimizer": {"type": "AdamW", "params": {"lr": 0.01}},
}
SMALL = dict(
    vocab_size=VOCAB,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=6,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    tie_word_embeddings=False,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)
FAMILIES = {
    "llama": {},
    "mistral": {},
    "qwen2": {},
    "qwen3": {},
    "gemma": {},
    "gemma2": {},
    "gemma3_text": {},
    "phi3": {},
    "olmo2": {},
    "granite": {},
    "cohere": {},
    "starcoder2": {},
    "stablelm": {},
    "glm4": {},
    "gpt_neox": {"rotary_pct": 0.5},
    "qwen3_moe": {"num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 32},
    "mixtral": {"num_local_experts": 4, "num_experts_per_tok": 2},
    "deepseek_v3": {
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "kv_lora_rank": 16,
        "q_lora_rank": 32,
        "qk_rope_head_dim": 8,
        "qk_nope_head_dim": 8,
        "v_head_dim": 16,
        "first_k_dense_replace": 1,
        "n_group": 1,
        "topk_group": 1,
        "head_dim": 8,
        "num_key_value_heads": 4,
    },
    "glm4_moe": {
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "first_k_dense_replace": 1,
        "n_group": 1,
        "topk_group": 1,
    },
    "qwen3_5_text": {
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
    },
    "gpt2": {
        "n_embd": 64,
        "n_layer": 6,
        "n_head": 4,
        "n_positions": 64,
        "resid_pdrop": 0.0,
        "embd_pdrop": 0.0,
        "attn_pdrop": 0.0,
    },
    "falcon": {"num_kv_heads": 2, "new_decoder_architecture": True, "head_dim": None},
    "bloom": {},
    "mamba": {"state_size": 8, "expand": 2},
}


def tiny(family):
    kwargs = {k: v for k, v in {**SMALL, **FAMILIES[family]}.items() if v is not None}
    try:
        cfg = transformers.AutoConfig.for_model(family, **kwargs)
    except ValueError:
        pytest.skip(f"transformers {transformers.__version__} has no {family}")
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return transformers.AutoModelForCausalLM.from_config(cfg).float()


def batches():
    g = torch.Generator().manual_seed(1)
    return [
        (ids, ids)
        for ids in (torch.randint(3, VOCAB, (ROWS, SEQ), generator=g) for _ in range(N_MB))
    ]


def lm_loss(logits, labels):
    return F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), labels[:, 1:].reshape(-1))


@pytest.fixture(scope="module")
def ray_ctx():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.pathsep.join([os.path.join(here, "..", "unit"), here])  # stub engine, lm_loss
    ray.init(
        num_cpus=4,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={"env_vars": {"PYTHONPATH": path}},
    )
    yield
    ray.shutdown()


@pytest.fixture()
def hf_stages(monkeypatch):
    """Real runtime and default stage builder choice; only the engine is the
    CPU stub."""
    from ray_deepspeed_pipeline import stage_group

    real = stage_group.create_stage_clients
    built = []

    def cpu_clients(model, plan, loss_fn, **kw):
        clients = real(
            model, plan, loss_fn, engine_factory=stub_engine_factory, use_gpu=False, **kw
        )
        built.append(clients)
        return clients

    monkeypatch.setattr(stage_group, "create_stage_clients", cpu_clients)
    yield
    for clients in built:
        for client in clients:
            client.shutdown()


@pytest.mark.parametrize("family", list(FAMILIES))
def test_family_trains_like_unsplit_model(family, ray_ctx, hf_stages):
    reference = tiny(family)
    opt = torch.optim.AdamW(reference.parameters(), lr=0.01)
    expected = []
    for _ in range(2):
        opt.zero_grad()
        losses = [
            lm_loss(reference(input_ids=x, use_cache=False).logits, y) / N_MB for x, y in batches()
        ]
        sum(losses).backward()
        opt.step()
        expected.append(float(sum(losses)))

    engine, _, _, _ = rdsp.initialize(
        model=tiny(family),
        config=DS,
        loss_fn=lm_loss,
        pipeline_config=rdsp.PipelineConfig(
            stages=3,
            partition=rdsp.ExplicitCuts((2, 4)),
            stage_overrides=(StageOverride(stage=1, num_gpus=2),),
        ),
    )
    got = [float(engine.train_batch(data_iter=iter(batches()))) for _ in range(2)]
    assert got == pytest.approx(expected, rel=1e-4)


def test_only_validated_types_keep_the_llama_stage():
    """With SDPA attention; under eager attention every family gets the HF
    builder, which is what the training test above runs."""

    def sdpa(family):
        model = tiny(family)
        model.config._attn_implementation = "sdpa"
        return model

    chosen = {family: select_stage_builder(sdpa(family)) for family in FAMILIES}
    assert {f for f, b in chosen.items() if b is build_causal_lm_stage} == {
        "llama",
        "qwen3",
        "qwen3_moe",
    }
    assert all(
        b is build_hf_stage for f, b in chosen.items() if f not in ("llama", "qwen3", "qwen3_moe")
    )


def tiny_glm5():
    """GLM-5.3 (glm5_next): 4-stream hyper-connected hidden state, sparse
    attention whose top-k indices each block hands the next, linear
    attention, MoE, a vision encoder in front."""
    config_mod = pytest.importorskip("transformers.models.glm5_next.configuration_glm5_next")
    text = dict(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=16,
        q_lora_rank=32,
        v_head_dim=16,
        qk_nope_head_dim=16,
        index_topk=8,
        index_kpool=4,
        index_head_dim=16,
        index_n_heads=2,
        linear_head_dim=16,
        linear_num_heads=4,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    cfg = config_mod.Glm5NextConfig(
        text_config=text,
        vision_config=dict(depth=2, hidden_size=32, num_heads=2, out_hidden_size=64),
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return transformers.Glm5NextForConditionalGeneration(cfg).float()


def test_glm5_blocks_handing_values_to_the_next_block(ray_ctx, hf_stages):
    reference = tiny_glm5()
    opt = torch.optim.AdamW(reference.parameters(), lr=0.01)
    expected = []
    for _ in range(2):
        opt.zero_grad()
        losses = [
            lm_loss(reference(input_ids=x, use_cache=False).logits, y) / N_MB for x, y in batches()
        ]
        sum(losses).backward()
        opt.step()
        expected.append(float(sum(losses)))

    engine, _, _, _ = rdsp.initialize(
        model=tiny_glm5(),
        config=DS,
        loss_fn=lm_loss,
        pipeline_config=rdsp.PipelineConfig(
            stages=3,
            partition=rdsp.ExplicitCuts((2, 4)),
            stage_overrides=(StageOverride(stage=1, num_gpus=2),),
        ),
    )
    got = [float(engine.train_batch(data_iter=iter(batches()))) for _ in range(2)]
    assert got == pytest.approx(expected, rel=1e-4)


def test_engine_gets_only_top_level_tensors():
    """Under AutoTP, DeepSpeed's first-forward check that TP ranks got the
    same inputs compares only top-level tensors; a nested dict makes it
    raise on some ranks and hang the rest."""
    from test_deepspeed_adapter import stub_engine_factory

    from ray_deepspeed_pipeline.deepspeed_adapter import DeepSpeedStageAdapter
    from ray_deepspeed_pipeline.partition import partition_parameters

    model = tiny("llama")
    parts = partition_parameters(model, rdsp.ExplicitCuts((3,)), 2)
    stages = [build_hf_stage(model, p.block_start, p.block_stop, p.parameter_names) for p in parts]
    calls = []

    def recording_factory(module, conf):
        engine = stub_engine_factory(module, conf)
        call = engine.__call__
        engine.__class__ = type(
            "Recording",
            (type(engine),),
            {"__call__": lambda self, *a, **k: (calls.append((a, k)), call(*a, **k))[1]},
        )
        return engine

    first = DeepSpeedStageAdapter(
        stages[0], {}, 1, is_first=True, is_last=False, engine_factory=recording_factory
    )
    last = DeepSpeedStageAdapter(
        stages[1],
        {},
        1,
        is_first=False,
        is_last=True,
        loss_fn=lm_loss,
        engine_factory=recording_factory,
    )
    ids = batches()[0][0]
    hidden, extras = first.forward(0, ids)
    last.forward(0, hidden, labels=ids, extras=extras)
    assert calls[1][1], "the second stage received block arguments"
    for args, kwargs in calls:
        assert all(torch.is_tensor(v) for v in (*args, *kwargs.values()))
