# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""A vision-language model (tiny Qwen3-VL) through rdsp.initialize() on real
Ray CPU actors, with only the DeepSpeed engine swapped for the torch stub.

Each pipelined run must reproduce the unsplit model's losses on the same
batches: images reach the vision encoder on stage 0, and the multimodal rotary
tables it computes travel to the later stages with the hidden state."""

import os

import pytest
import torch
import torch.nn.functional as F
from test_deepspeed_adapter import stub_engine_factory

import ray_deepspeed_pipeline as rdsp
from ray_deepspeed_pipeline.config import StageOverride
from ray_deepspeed_pipeline.errors import ValidationError

ray = pytest.importorskip("ray")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
    pytest.skip("transformers without Qwen3-VL", allow_module_level=True)

VOCAB, SEQ, ROWS, N_MB = 256, 12, 2, 2
IMAGE, IMAGE_PAD, VISION_START = 250, 250, 252
GRID = (1, 4, 4)  # 16 patches, 4 tokens after 2x2 merging
PATCH_DIM = 3 * 1 * 4 * 4
DS = {
    "train_batch_size": N_MB * ROWS,
    "gradient_accumulation_steps": N_MB,
    "optimizer": {"type": "AdamW", "params": {"lr": 0.01}},
}


def tiny_qwen3_vl():
    cfg = transformers.Qwen3VLConfig(
        text_config=dict(
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            rope_scaling={
                "rope_type": "default",
                "mrope_section": [2, 3, 3],
                "mrope_interleaved": True,
            },
        ),
        vision_config=dict(
            depth=4,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=64,
            patch_size=4,
            spatial_merge_size=2,
            temporal_patch_size=1,
            deepstack_visual_indexes=[1, 2],
            num_position_embeddings=16,
        ),
        image_token_id=IMAGE_PAD,
        video_token_id=251,
        vision_start_token_id=VISION_START,
        tie_word_embeddings=False,
    )
    cfg._attn_implementation = "sdpa"
    return cfg


def tiny_qwen3_5_vl(tied=False):
    """Qwen3.5: linear-attention (Gated DeltaNet) and full-attention layers,
    and a vision encoder with no deepstack, so nothing it computes enters the
    decoder past the input embeddings."""
    cfg = transformers.Qwen3_5Config(
        text_config=dict(
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            layer_types=["linear_attention", "linear_attention", "full_attention"] * 2,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
                "partial_rotary_factor": 0.5,
            },
        ),
        vision_config=dict(
            depth=4,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=64,
            patch_size=4,
            spatial_merge_size=2,
            temporal_patch_size=1,
            num_position_embeddings=16,
        ),
        image_token_id=IMAGE_PAD,
        video_token_id=251,
        vision_start_token_id=VISION_START,
        tie_word_embeddings=tied,
    )
    cfg.get_text_config().tie_word_embeddings = tied
    cfg._attn_implementation = "sdpa"
    return cfg


def tiny_qwen3_5_moe_vl():
    """Qwen3.5-MoE: the Qwen3.5 layers with every MLP a mixture of experts
    (routed experts, top-k, plus a gated shared expert)."""
    cfg = transformers.Qwen3_5MoeConfig(
        text_config=dict(
            vocab_size=VOCAB,
            hidden_size=64,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
            num_experts=8,
            num_experts_per_tok=2,
            layer_types=["linear_attention", "linear_attention", "full_attention"] * 2,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
                "partial_rotary_factor": 0.5,
            },
        ),
        vision_config=dict(
            depth=4,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=64,
            patch_size=4,
            spatial_merge_size=2,
            temporal_patch_size=1,
            num_position_embeddings=16,
        ),
        image_token_id=IMAGE_PAD,
        video_token_id=251,
        vision_start_token_id=VISION_START,
        tie_word_embeddings=False,
    )
    cfg._attn_implementation = "sdpa"
    return cfg


def make_batches(seed=0):
    """N_MB microbatches of ROWS rows, one image per row at the same place, so
    every row has the same number of masked label positions."""
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(N_MB):
        ids = torch.randint(0, 200, (ROWS, SEQ), generator=g)
        ids[:, 2] = VISION_START
        ids[:, 3:7] = IMAGE_PAD
        labels = ids.clone()
        labels[:, 2:7] = -100
        inputs = {
            "input_ids": ids,
            "mm_token_type_ids": (ids == IMAGE_PAD).long(),
            # not row-shaped: one packed tensor per row, concatenated per rank
            "pixel_values": [torch.randn(16, PATCH_DIM, generator=g) for _ in range(ROWS)],
            "image_grid_thw": [torch.tensor([GRID]) for _ in range(ROWS)],
        }
        batches.append((inputs, labels))
    return batches


def lm_loss(logits, labels):
    return F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), labels[:, 1:].reshape(-1))


def unsplit_losses(model, batches, steps, config=DS):
    """Loss per step of the whole model in one process, same optimizer."""
    opt = config["optimizer"]
    kind = torch.optim.SGD if opt["type"] == "SGD" else torch.optim.AdamW
    opt = kind(model.parameters(), **opt["params"])
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        total = 0.0
        for inputs, labels in batches:
            full = {k: torch.cat(v) if isinstance(v, list) else v for k, v in inputs.items()}
            loss = lm_loss(model(**full, use_cache=False).logits, labels) / len(batches)
            loss.backward()
            total += float(loss)
        opt.step()
        losses.append(total)
    return losses


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
def stub_engines(monkeypatch):
    """Real runtime and default stage builder; only the engine is the CPU stub."""
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


def pipelined_losses(
    model,
    cuts,
    steps,
    overrides=(),
    weights=None,
    colocated_vision=None,
    evaluate=False,
    config=DS,
    prefetch=False,
):
    engine, _, _, _ = rdsp.initialize(
        model=model,
        config=config,
        loss_fn=lm_loss,
        weights=weights,
        pipeline_config=rdsp.PipelineConfig(
            stages=len(cuts) + 1,
            partition=rdsp.ExplicitCuts(cuts),
            stage_overrides=overrides,
            colocated_vision=colocated_vision,
            prefetch=prefetch,
        ),
    )
    run = engine.eval_batch if evaluate else engine.train_batch
    if prefetch:  # one iterator for the whole run
        data = iter([b for _ in range(steps) for b in make_batches()])
        return [float(run(data_iter=data)) for _ in range(steps)]
    return [float(run(data_iter=iter(make_batches()))) for _ in range(steps)]


def test_two_stages_match_unsplit_model(ray_ctx, stub_engines):
    torch.manual_seed(0)
    model = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    reference = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    reference.load_state_dict(model.state_dict())

    expected = unsplit_losses(reference, make_batches(), steps=3)
    assert pipelined_losses(model, (3,), steps=3) == pytest.approx(expected, rel=1e-4)


def test_data_parallel_vision_stage_matches_unsplit_model(ray_ctx, stub_engines):
    """Stage 0 (vision encoder, blocks 0-1) on 2 data-parallel ranks: each
    rank gets its rows' images, and the next stage reassembles the rows of
    both the hidden state and the rotary tables."""
    torch.manual_seed(1)
    model = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    reference = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    reference.load_state_dict(model.state_dict())

    expected = unsplit_losses(reference, make_batches(), steps=2)
    got = pipelined_losses(model, (2, 4), steps=2, overrides=(StageOverride(stage=0, num_gpus=2),))
    assert got == pytest.approx(expected, rel=1e-4)


def qwen3_5_vl_pair(seed, tied=False):
    """A tiny Qwen3.5-VL and an identical copy for the unsplit reference."""
    torch.manual_seed(seed)
    model = transformers.Qwen3_5ForConditionalGeneration(tiny_qwen3_5_vl(tied)).float()
    reference = transformers.Qwen3_5ForConditionalGeneration(tiny_qwen3_5_vl(tied)).float()
    reference.load_state_dict(model.state_dict())
    return model, reference


@pytest.mark.skipif(
    not hasattr(transformers, "Qwen3_5ForConditionalGeneration"),
    reason="transformers without Qwen3.5",
)
def test_qwen3_5_vl_with_data_parallel_vision_stage_matches_unsplit_model(ray_ctx, stub_engines):
    model, reference = qwen3_5_vl_pair(seed=6)
    expected = unsplit_losses(reference, make_batches(), steps=2)
    got = pipelined_losses(model, (1, 4), steps=2, overrides=(StageOverride(stage=0, num_gpus=2),))
    assert got == pytest.approx(expected, rel=1e-4)


@pytest.mark.skipif(
    not hasattr(transformers, "Qwen3_5MoeForConditionalGeneration"),
    reason="transformers without Qwen3.5-MoE",
)
@pytest.mark.parametrize(
    "cuts,overrides",
    [
        ((3,), ()),
        ((1, 4), (StageOverride(stage=1, num_gpus=2),)),  # experts on 2 data-parallel ranks
    ],
    ids=["two-stages", "data-parallel-experts"],
)
def test_qwen3_5_moe_vl_matches_unsplit_model(ray_ctx, stub_engines, cuts, overrides):
    torch.manual_seed(15)
    model = transformers.Qwen3_5MoeForConditionalGeneration(tiny_qwen3_5_moe_vl()).float()
    reference = transformers.Qwen3_5MoeForConditionalGeneration(tiny_qwen3_5_moe_vl()).float()
    reference.load_state_dict(model.state_dict())
    expected = unsplit_losses(reference, make_batches(), steps=2)
    assert pipelined_losses(model, cuts, steps=2, overrides=overrides) == pytest.approx(
        expected, rel=1e-4
    )


@pytest.mark.skipif(
    not hasattr(transformers, "Qwen3_5MoeForConditionalGeneration"),
    reason="transformers without Qwen3.5-MoE",
)
def test_vision_stage_measured_then_re_cut_matches_unsplit_model(ray_ctx, stub_engines):
    """--cuts 0,measured: the vision encoder alone on stage 0 (2 ranks), the decoder over two
    stages; measure_stages times the stages, the decoder is re-cut by what they measured, and
    the re-cut pipeline (vision stage unchanged) trains as the unsplit model does."""
    import dataclasses

    torch.manual_seed(16)
    model = transformers.Qwen3_5MoeForConditionalGeneration(tiny_qwen3_5_moe_vl()).float()
    reference = transformers.Qwen3_5MoeForConditionalGeneration(tiny_qwen3_5_moe_vl()).float()
    reference.load_state_dict(model.state_dict())
    overrides = (StageOverride(stage=0, num_gpus=2),)

    def build(policy):
        engine, _, _, _ = rdsp.initialize(
            model=model,
            config=DS,
            loss_fn=lm_loss,
            pipeline_config=rdsp.PipelineConfig(
                stages=3, partition=policy, stage_overrides=overrides
            ),
        )
        return engine

    estimate = rdsp.BalancedTransformerBlocks(vision_stage=True)
    calibration = build(estimate)
    measured = calibration.measure_stages(
        (b for _ in range(3) for b in make_batches()), steps=3, skip=1
    )
    calibration.shutdown()
    assert measured.cuts[0] == 0 and len(measured.seconds) == 3

    engine = build(dataclasses.replace(estimate, measured=measured))
    assert engine.stage_blocks[0] == (0, 0)
    expected = unsplit_losses(reference, make_batches(), steps=2)
    got = [float(engine.train_batch(data_iter=iter(make_batches()))) for _ in range(2)]
    assert got == pytest.approx(expected, rel=1e-4)


@pytest.mark.skipif(
    not hasattr(transformers, "Qwen3_5ForConditionalGeneration"),
    reason="transformers without Qwen3.5",
)
def test_vision_only_first_stage_matches_unsplit_model(ray_ctx, stub_engines):
    """A first cut at 0 gives the vision encoder a stage of its own (here on
    2 data-parallel ranks) holding nothing else: it sends the image
    features at the image positions, the token ids and the rotary tables;
    the next stage owns the embeddings and embeds the text tokens itself."""
    from ray_deepspeed_pipeline.partition import partition_parameters

    model, reference = qwen3_5_vl_pair(seed=7)
    first, second, _ = partition_parameters(model, rdsp.ExplicitCuts((0, 3)), 3)
    assert all(".visual." in n for n in first.parameter_names)
    assert any("embed_tokens" in n for n in second.parameter_names)

    expected = unsplit_losses(reference, make_batches(), steps=2)
    got = pipelined_losses(model, (0, 3), steps=2, overrides=(StageOverride(stage=0, num_gpus=2),))
    assert got == pytest.approx(expected, rel=1e-4)


@pytest.mark.skipif(
    not hasattr(transformers, "Qwen3_5ForConditionalGeneration"),
    reason="transformers without Qwen3.5",
)
def test_tied_embeddings_on_one_stage_train_as_one_matrix(ray_ctx, stub_engines):
    """With the vision encoder on a stage of its own, the next stage holds
    both the input embedding and the output head; a model that ties them
    keeps one matrix there, trained by both uses, as unsplit."""
    model, reference = qwen3_5_vl_pair(seed=9, tied=True)
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    expected = unsplit_losses(reference, make_batches(), steps=3)
    assert pipelined_losses(model, (0,), steps=3) == pytest.approx(expected, rel=1e-4)


needs_qwen3_5 = pytest.mark.skipif(
    not hasattr(transformers, "Qwen3_5ForConditionalGeneration"),
    reason="transformers without Qwen3.5",
)


@needs_qwen3_5
@pytest.mark.parametrize(
    "cuts,overrides",
    [
        ((3,), (StageOverride(stage=0, num_gpus=2),)),  # 3 vision ranks, 4 images
        ((2,), (StageOverride(stage=1, num_gpus=2),)),  # images reach stage 0 from stage 1
    ],
    ids=["dp-first-stage", "dp-last-stage"],
)
@pytest.mark.parametrize("per_microbatch", [False, True], ids=["whole-step", "per-microbatch"])
def test_colocated_vision_matches_unsplit_model(
    ray_ctx, stub_engines, cuts, overrides, per_microbatch
):
    """Colocated vision: every rank of every stage hosts the vision encoder
    and encodes a share of the step's images; stage 0 gets their features
    in place of pixels, and the feature gradients go back to the rank that
    encoded each image."""
    model, reference = qwen3_5_vl_pair(seed=8)
    expected = unsplit_losses(reference, make_batches(), steps=3)
    got = pipelined_losses(
        model,
        cuts,
        steps=3,
        overrides=overrides,
        colocated_vision=rdsp.ColocatedVision(encode_per_microbatch=per_microbatch),
    )
    assert got == pytest.approx(expected, rel=1e-4)


def test_prefetch_matches_unsplit_model(ray_ctx, stub_engines):
    """Reading the next step ahead changes when data moves, not what trains."""
    torch.manual_seed(0)
    model = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    reference = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    reference.load_state_dict(model.state_dict())

    expected = unsplit_losses(reference, make_batches(), steps=3)
    got = pipelined_losses(
        model, (3,), steps=3, prefetch=True, overrides=(StageOverride(stage=1, num_gpus=2),)
    )
    assert got == pytest.approx(expected, rel=1e-4)


@needs_qwen3_5
def test_prefetch_with_colocated_vision_matches_unsplit_model(ray_ctx, stub_engines):
    model, reference = qwen3_5_vl_pair(seed=8)
    expected = unsplit_losses(reference, make_batches(), steps=3)
    got = pipelined_losses(
        model,
        (2,),
        steps=3,
        prefetch=True,
        overrides=(StageOverride(stage=1, num_gpus=2),),
        colocated_vision=rdsp.ColocatedVision(),
    )
    assert got == pytest.approx(expected, rel=1e-4)


def token_sum(logits, labels):
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, VOCAB),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )


def token_count(labels):
    return int((labels[:, 1:] != -100).sum())


def uneven_batches():
    """Rows whose captions differ in length: a mean of per-rank means is
    then not the mean over the step's tokens."""
    batches = make_batches()
    for i, (_, labels) in enumerate(batches):
        labels[0, 7 + 2 * i :] = -100
    return batches


@needs_qwen3_5
@pytest.mark.parametrize("colocated", [False, True], ids=["stages", "colocated"])
def test_token_mean_loss_matches_unsplit_model_on_uneven_rows(ray_ctx, stub_engines, colocated):
    """TokenMeanLoss: the step's loss is its summed token loss over all
    microbatches and data-parallel ranks divided by its token count, the
    same objective the unsplit model sees, with every row weighted by its
    tokens."""
    sgd = dict(DS, optimizer={"type": "SGD", "params": {"lr": 0.5, "momentum": 0.9}})
    model, reference = qwen3_5_vl_pair(seed=14)
    opt = torch.optim.SGD(reference.parameters(), lr=0.5, momentum=0.9)
    expected = []
    for _ in range(3):
        opt.zero_grad()
        batches = uneven_batches()
        total = sum(token_count(labels) for _, labels in batches)
        step = 0.0
        for inputs, labels in batches:
            full = {k: torch.cat(v) if isinstance(v, list) else v for k, v in inputs.items()}
            loss = token_sum(reference(**full, use_cache=False).logits, labels) / total
            loss.backward()
            step += float(loss)
        opt.step()
        expected.append(step)

    engine, _, _, _ = rdsp.initialize(
        model=model,
        config=sgd,
        loss_fn=rdsp.TokenMeanLoss(token_sum, token_count),
        pipeline_config=rdsp.PipelineConfig(
            stages=2,
            partition=rdsp.ExplicitCuts((3,)),
            stage_overrides=(StageOverride(stage=1, num_gpus=2),),
            colocated_vision=rdsp.ColocatedVision() if colocated else None,
        ),
    )
    got = [float(engine.train_batch(data_iter=iter(uneven_batches()))) for _ in range(3)]
    assert got == pytest.approx(expected, rel=1e-4)


@needs_qwen3_5
def test_per_microbatch_training_still_reports_the_token_mean(ray_ctx, stub_engines):
    """TokenMeanLoss(per_microbatch=True): each rank's microbatch trains on
    its own token mean (here one row each: every row counts the same, as in
    Megatron without per-token loss), while the step loss reported is still
    the step's token mean, as Megatron logs it."""
    sgd = dict(DS, optimizer={"type": "SGD", "params": {"lr": 0.5, "momentum": 0.9}})
    model, reference = qwen3_5_vl_pair(seed=15)
    opt = torch.optim.SGD(reference.parameters(), lr=0.5, momentum=0.9)
    expected = []
    for _ in range(3):
        opt.zero_grad()
        batches = uneven_batches()
        total = sum(token_count(labels) for _, labels in batches)
        rows = sum(len(labels) for _, labels in batches)
        step = 0.0
        for inputs, labels in batches:
            full = {k: torch.cat(v) if isinstance(v, list) else v for k, v in inputs.items()}
            logits = reference(**full, use_cache=False).logits
            for r in range(len(labels)):  # one row per data-parallel rank
                row = token_sum(logits[r : r + 1], labels[r : r + 1])
                (row / token_count(labels[r : r + 1]) / rows).backward(retain_graph=True)
                step += float(row) / total
        opt.step()
        expected.append(step)

    engine, _, _, _ = rdsp.initialize(
        model=model,
        config=sgd,
        loss_fn=rdsp.TokenMeanLoss(token_sum, token_count, per_microbatch=True),
        pipeline_config=rdsp.PipelineConfig(
            stages=2,
            partition=rdsp.ExplicitCuts((3,)),
            stage_overrides=(StageOverride(stage=1, num_gpus=2),),
        ),
    )
    got = [float(engine.train_batch(data_iter=iter(uneven_batches()))) for _ in range(3)]
    assert got == pytest.approx(expected, rel=1e-4)


@needs_qwen3_5
def test_colocated_vision_gradients_have_the_right_scale(ray_ctx, stub_engines):
    """SGD, unlike Adam, moves a parameter in proportion to its gradient, so
    a mis-scaled encoder gradient (the first stage's data-parallel degree,
    the number of encoding ranks) changes the losses."""
    sgd = dict(DS, optimizer={"type": "SGD", "params": {"lr": 0.5, "momentum": 0.9}})
    model, reference = qwen3_5_vl_pair(seed=13)
    expected = unsplit_losses(reference, make_batches(), steps=3, config=sgd)
    got = pipelined_losses(
        model,
        (3,),
        steps=3,
        config=sgd,
        overrides=(StageOverride(stage=0, num_gpus=2),),
        colocated_vision=rdsp.ColocatedVision(),
    )
    assert got == pytest.approx(expected, rel=1e-4)


@needs_qwen3_5
def test_colocated_vision_first_stage_encodes_its_first_images(ray_ctx, stub_engines):
    """With enough images per step the first stage encodes its first
    microbatches' images itself and the other stages the rest, features
    and gradients crossing as each microbatch needs them."""
    batches = make_batches(0) + make_batches(1)  # 4 microbatches, 8 images
    config = dict(DS, train_batch_size=8, gradient_accumulation_steps=4)
    model, reference = qwen3_5_vl_pair(seed=15)
    expected = unsplit_losses(reference, batches, steps=2, config=config)
    engine, _, _, _ = rdsp.initialize(
        model=model,
        config=config,
        loss_fn=lm_loss,
        pipeline_config=rdsp.PipelineConfig(
            stages=2,
            partition=rdsp.ExplicitCuts((3,)),
            stage_overrides=(StageOverride(stage=0, num_gpus=2),),
            colocated_vision=rdsp.ColocatedVision(),
        ),
    )
    got = [float(engine.train_batch(data_iter=iter(batches))) for _ in range(2)]
    assert got == pytest.approx(expected, rel=1e-4)


@needs_qwen3_5
def test_colocated_vision_leaves_the_encoder_off_every_stage():
    from ray_deepspeed_pipeline.compiler import lower

    model = transformers.Qwen3_5ForConditionalGeneration(tiny_qwen3_5_vl())
    plan = lower(
        model,
        rdsp.PipelineConfig(
            stages=2, partition=rdsp.ExplicitCuts((3,)), colocated_vision=rdsp.ColocatedVision()
        ),
        DS,
    )
    assert plan.colocated_vision.module == "model.visual"
    staged = {n for s in plan.stages for n in s.parameter_names}
    assert staged.isdisjoint(plan.colocated_vision.parameter_names)
    assert staged | set(plan.colocated_vision.parameter_names) == {
        n for n, _ in model.named_parameters()
    }


@needs_qwen3_5
def test_colocated_vision_checkpoint_resumes_the_encoder(ray_ctx, stub_engines, tmp_path):
    """The encoder's weights and optimizer state round-trip: a run resumed
    from a checkpoint continues with the same losses."""

    def make_engine(seed):
        model, _ = qwen3_5_vl_pair(seed)
        engine, _, _, _ = rdsp.initialize(
            model=model,
            config=DS,
            loss_fn=lm_loss,
            pipeline_config=rdsp.PipelineConfig(
                stages=2,
                partition=rdsp.ExplicitCuts((3,)),
                stage_overrides=(StageOverride(stage=0, num_gpus=2),),
                colocated_vision=rdsp.ColocatedVision(),
            ),
        )
        return engine

    a = make_engine(seed=11)
    a.train_batch(data_iter=iter(make_batches()))
    a.save_checkpoint(str(tmp_path), "c1")
    continued = [float(a.train_batch(data_iter=iter(make_batches()))) for _ in range(2)]
    for worker in a._coordinator._workers:  # the test cluster fits one pipeline
        worker.shutdown()
    b = make_engine(seed=12)  # different initial weights, encoder included
    b.load_checkpoint(str(tmp_path), "c1")
    resumed = [float(b.train_batch(data_iter=iter(make_batches()))) for _ in range(2)]
    assert resumed == pytest.approx(continued, rel=1e-5)


@needs_qwen3_5
def test_colocated_vision_with_recompute_and_eval(ray_ctx, stub_engines):
    model, reference = qwen3_5_vl_pair(seed=9)
    inputs = [(inp, labels) for inp, labels in make_batches()]
    with torch.no_grad():
        expected = sum(
            float(
                lm_loss(
                    reference(
                        **{k: torch.cat(v) if isinstance(v, list) else v for k, v in inp.items()},
                        use_cache=False,
                    ).logits,
                    labels,
                )
            )
            for inp, labels in inputs
        ) / len(inputs)
    got = pipelined_losses(
        model, (3,), steps=1, evaluate=True, colocated_vision=rdsp.ColocatedVision(recompute=True)
    )
    assert got == pytest.approx([expected], rel=1e-4)


@needs_qwen3_5
def test_colocated_vision_loads_its_weights_from_the_checkpoint(ray_ctx, stub_engines, tmp_path):
    accelerate = pytest.importorskip("accelerate")
    trained, _ = qwen3_5_vl_pair(seed=10)
    trained.save_pretrained(tmp_path)
    expected = unsplit_losses(trained, make_batches(), steps=2)
    with accelerate.init_empty_weights():
        skeleton = transformers.Qwen3_5ForConditionalGeneration(tiny_qwen3_5_vl()).float()
    got = pipelined_losses(
        skeleton, (3,), steps=2, weights=str(tmp_path), colocated_vision=rdsp.ColocatedVision()
    )
    assert got == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize(
    "make,cuts,match",
    [
        (
            lambda: transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()),
            (3,),
            "inside the decoder",
        ),
        (
            lambda: transformers.Qwen3_5ForConditionalGeneration(tiny_qwen3_5_vl()),
            (0, 3),
            "vision-only",
        ),
        (
            lambda: transformers.Qwen3ForCausalLM(
                transformers.Qwen3Config(
                    vocab_size=VOCAB,
                    hidden_size=64,
                    intermediate_size=128,
                    num_hidden_layers=4,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=16,
                )
            ),
            (2,),
            "vision encoder",
        ),
    ],
    ids=["deepstack", "vision-only-stage", "text-model"],
)
def test_colocated_vision_rejected_where_it_cannot_apply(make, cuts, match):
    from ray_deepspeed_pipeline.compiler import lower

    if not hasattr(transformers, "Qwen3_5ForConditionalGeneration"):
        pytest.skip("transformers without Qwen3.5")
    with pytest.raises(ValidationError, match=match):
        lower(
            make(),
            rdsp.PipelineConfig(
                stages=len(cuts) + 1,
                partition=rdsp.ExplicitCuts(cuts),
                colocated_vision=rdsp.ColocatedVision(),
            ),
            DS,
        )


@needs_qwen3_5
def test_balanced_vision_stage_puts_the_encoder_alone_first_and_balances_the_rest():
    """BalancedTransformerBlocks(vision_stage=True): stage 0 holds the vision
    encoder and no decoder block; the decoder blocks are split over the other
    stages exactly as a balanced split of those stages alone would split them."""
    from ray_deepspeed_pipeline.partition import (
        _balanced_cuts,
        _compute_costs,
        find_block_list,
        partition_parameters,
    )

    model = transformers.Qwen3_5ForConditionalGeneration(tiny_qwen3_5_vl())
    parts = partition_parameters(model, rdsp.BalancedTransformerBlocks(vision_stage=True), 3)
    assert (parts[0].block_start, parts[0].block_stop) == (0, 0)
    assert any(n.startswith("model.visual.") for n in parts[0].parameter_names)
    name, blocks = find_block_list(model)
    _, per_block, post = _compute_costs(model, name, blocks)
    expected = _balanced_cuts((0, per_block, post), 2, 0, (1, 1))
    assert [p.block_start for p in parts[1:]] == [0, *expected]
    assert parts[-1].block_stop == len(blocks)


def test_balanced_vision_stage_needs_a_vision_encoder():
    from ray_deepspeed_pipeline.partition import partition_parameters

    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
        )
    )
    with pytest.raises(ValidationError, match="vision encoder"):
        partition_parameters(model, rdsp.BalancedTransformerBlocks(vision_stage=True), 2)


@needs_qwen3_5
def test_colocated_vision_rejects_a_stage_compile_vision_it_would_ignore():
    """compile_vision compiles the encoder a stage holds; a colocated encoder is
    on no stage, so the flag would do nothing. ColocatedVision(compile=) is the
    switch that applies."""
    from ray_deepspeed_pipeline.compiler import lower

    model = transformers.Qwen3_5ForConditionalGeneration(tiny_qwen3_5_vl())
    with pytest.raises(ValidationError, match=r"ColocatedVision\(compile=True\)"):
        lower(
            model,
            rdsp.PipelineConfig(
                stages=2,
                partition=rdsp.ExplicitCuts((3,)),
                stage_overrides=(rdsp.StageOverride(stage=0, compile_vision=True),),
                colocated_vision=rdsp.ColocatedVision(),
            ),
            DS,
        )


@needs_qwen3_5
def test_block_list_is_the_decoders_even_when_the_encoder_is_as_deep():
    """Qwen3.5-2B: 24 vision blocks and 24 decoder layers. The pipeline's
    blocks are the decoder's; the vision encoder is never cut."""
    from ray_deepspeed_pipeline.partition import find_block_list

    cfg = tiny_qwen3_5_vl()
    cfg.vision_config.depth = cfg.text_config.num_hidden_layers
    name, blocks = find_block_list(transformers.Qwen3_5ForConditionalGeneration(cfg))
    assert name == "model.language_model.layers" and len(blocks) == 6


def test_first_cut_at_zero_needs_a_vision_encoder():
    """Without a vision encoder, a first stage without blocks would only
    look up embeddings."""
    from ray_deepspeed_pipeline.partition import partition_parameters

    text = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
        )
    )
    with pytest.raises(ValidationError, match="vision encoder"):
        partition_parameters(text, rdsp.ExplicitCuts((0, 2)), 3)


def test_meta_skeleton_loads_weights_per_stage(ray_ctx, stub_engines, tmp_path):
    """The driver holds only a meta-device skeleton; each stage reads its own
    tensors from the checkpoint directory and trains like the real model."""
    accelerate = pytest.importorskip("accelerate")
    torch.manual_seed(2)
    trained = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    trained.save_pretrained(tmp_path)
    expected = unsplit_losses(trained, make_batches(), steps=2)

    with accelerate.init_empty_weights():
        skeleton = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    assert all(p.is_meta for p in skeleton.parameters())
    got = pipelined_losses(skeleton, (3,), steps=2, weights=str(tmp_path))
    assert got == pytest.approx(expected, rel=1e-4)


def test_meta_skeleton_without_weights_is_rejected(ray_ctx, stub_engines):
    accelerate = pytest.importorskip("accelerate")
    with accelerate.init_empty_weights():
        skeleton = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl())
    with pytest.raises(Exception, match="meta device"):
        pipelined_losses(skeleton, (3,), steps=1)


def test_first_cut_before_vision_injection_is_rejected(ray_ctx, stub_engines):
    """Qwen3-VL adds vision features inside blocks 0-1 (deepstack), so those
    must stay on the first stage."""
    model = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl())
    with pytest.raises(ValidationError, match="first cut at 2"):
        pipelined_losses(model, (1,), steps=1)


def test_recompute_on_every_stage_matches_unsplit_model(ray_ctx, stub_engines):
    """Blocks rerun their forward during backward; the injected rotary tables
    and hidden state must be the ones the first run used."""
    torch.manual_seed(3)
    model = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    reference = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    reference.load_state_dict(model.state_dict())

    expected = unsplit_losses(reference, make_batches(), steps=2)
    got = pipelined_losses(
        model,
        (2, 4),
        steps=2,
        overrides=(
            StageOverride(stage=0, num_gpus=2, recompute=True),
            StageOverride(stage=1, recompute=True),
            StageOverride(stage=2, recompute=True),
        ),
    )
    assert got == pytest.approx(expected, rel=1e-4)


def test_balanced_cuts_keep_vision_injection_on_the_first_stage():
    from ray_deepspeed_pipeline.config import BalancedTransformerBlocks
    from ray_deepspeed_pipeline.partition import partition_parameters

    model = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl())
    for stages in (2, 3, 4):
        parts = partition_parameters(model, BalancedTransformerBlocks(), stages)
        assert parts[0].block_stop >= 2  # blocks 0-1 receive deepstack features


def test_hf_stage_carries_what_intra_stage_parallelism_needs():
    """AutoTP reads the text model's colwise/rowwise plan; Ulysses and AutoEP
    read head counts and MoE settings from the text config."""
    from ray_deepspeed_pipeline.hf_stage import build_hf_stage
    from ray_deepspeed_pipeline.partition import partition_parameters

    model = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl())
    part = partition_parameters(model, rdsp.ExplicitCuts((3,)), 2)[1]
    stage = build_hf_stage(model, part.block_start, part.block_stop, part.parameter_names)
    assert stage._tp_plan["layers.*.self_attn.q_proj"] == "colwise"
    assert stage._tp_plan["layers.*.self_attn.o_proj"] == "rowwise"
    assert stage._rdsp_hf_config.num_attention_heads == 4


@pytest.mark.skipif(
    not hasattr(transformers, "Qwen3_5ForConditionalGeneration"),
    reason="transformers without Qwen3.5",
)
def test_tensor_parallel_rules_cover_the_vision_encoder_and_linear_attention():
    """Under AutoTP the first stage splits its vision blocks too (fused qkv
    by thirds), not only the decoder; linear-attention projections split
    their output and gather it. The patch merger stays whole."""
    autotp = pytest.importorskip("deepspeed.module_inject.autotp_config")
    from ray_deepspeed_pipeline.deepspeed_adapter import _tp_partition_config
    from ray_deepspeed_pipeline.hf_stage import build_hf_stage
    from ray_deepspeed_pipeline.partition import partition_parameters

    model = transformers.Qwen3_5ForConditionalGeneration(tiny_qwen3_5_vl())
    part = partition_parameters(model, rdsp.ExplicitCuts((3,)), 2)[0]
    stage = build_hf_stage(model, 0, 3, part.parameter_names)
    rules = autotp.AutoTPConfig.from_dict(_tp_partition_config(stage))

    def spec(name):
        found = rules.find_matching_spec(f"model.{name}.weight", None)
        if found is None:
            return None
        return (found.partition_type.value, found.shape, found.gather_output)

    visual = "model.visual.blocks.1"
    assert spec(f"{visual}.attn.qkv") == ("column", (3, -1), False)
    assert spec(f"{visual}.attn.proj") == ("row", None, False)
    assert spec(f"{visual}.mlp.linear_fc1") == ("column", None, False)
    assert spec(f"{visual}.mlp.linear_fc2") == ("row", None, False)
    assert spec("model.visual.merger.linear_fc1") is None
    text = "model.language_model.layers"
    assert spec(f"{text}.0.linear_attn.in_proj_qkv") == ("column", None, True)
    assert spec(f"{text}.2.self_attn.q_proj") == ("column", None, False)
    assert spec(f"{text}.2.mlp.down_proj") == ("row", None, False)


@needs_qwen3_5
def test_gathered_projections_stay_whole_where_autotp_cannot_gather(monkeypatch):
    """A DeepSpeed whose AutoTP has no gather_output would split those
    projections and silently skip the gather; they stay whole instead."""
    from ray_deepspeed_pipeline import deepspeed_adapter
    from ray_deepspeed_pipeline.hf_stage import build_hf_stage
    from ray_deepspeed_pipeline.partition import partition_parameters

    model = transformers.Qwen3_5ForConditionalGeneration(tiny_qwen3_5_vl())
    part = partition_parameters(model, rdsp.ExplicitCuts((3,)), 2)[0]
    stage = build_hf_stage(model, 0, 3, part.parameter_names)
    monkeypatch.setattr(deepspeed_adapter, "_autotp_can_gather", lambda: False)
    specs = deepspeed_adapter._tp_partition_config(stage)["layer_specs"]
    assert not any("linear_attn" in p for spec in specs for p in spec["patterns"])
    assert any("self_attn" in p for spec in specs for p in spec["patterns"])


@needs_qwen3_5
def test_tensor_parallel_resizes_only_modules_holding_split_projections():
    """AutoTP divides size attributes (num_heads, embed_dim, hidden_size) of
    the modules it walks; some DeepSpeed versions do so even where nothing
    was split. The patch embedding and merger read theirs in forward, so
    only modules holding a split projection may be resized."""
    from ray_deepspeed_pipeline.deepspeed_adapter import (
        _keep_unsplit_module_sizes,
        _tp_partition_config,
    )
    from ray_deepspeed_pipeline.hf_stage import build_hf_stage
    from ray_deepspeed_pipeline.partition import partition_parameters

    model = transformers.Qwen3_5ForConditionalGeneration(tiny_qwen3_5_vl())
    part = partition_parameters(model, rdsp.ExplicitCuts((3,)), 2)[0]
    stage = build_hf_stage(model, 0, 3, part.parameter_names)
    _keep_unsplit_module_sizes(stage, _tp_partition_config(stage))
    visual = stage.model.model.visual
    assert visual.patch_embed.replaced and visual.merger.replaced
    assert not getattr(visual.blocks[0].attn, "replaced", False)  # num_heads must shrink
    assert not getattr(stage.model.model.language_model.layers[2].self_attn, "replaced", False)


def test_sequence_parallel_first_stage_of_a_vision_model_rejected():
    """Image positions depend on the whole sequence, which a sequence shard
    of stage 0 does not have."""
    from ray_deepspeed_pipeline.compiler import lower

    model = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl())
    with pytest.raises(ValidationError, match="sequence parallelism"):
        lower(
            model,
            rdsp.PipelineConfig(
                stages=2,
                partition=rdsp.ExplicitCuts((3,)),
                stage_overrides=(StageOverride(stage=0, num_gpus=2, sp=2),),
            ),
            DS,
        )


def test_tied_checkpoint_loads_into_an_untied_skeleton(ray_ctx, stub_engines, tmp_path):
    """Qwen3-VL-2B ties its embeddings, so its checkpoint has no lm_head.
    rdsp needs them untied (they sit on different stages); the head is then
    loaded from the embedding the model's tied-weights map names."""
    accelerate = pytest.importorskip("accelerate")
    tied_cfg = tiny_qwen3_vl()
    tied_cfg.tie_word_embeddings = tied_cfg.text_config.tie_word_embeddings = True
    torch.manual_seed(4)
    tied = transformers.Qwen3VLForConditionalGeneration(tied_cfg).float()
    tied.save_pretrained(tmp_path)
    expected = unsplit_losses(tied, make_batches(), steps=1)

    with accelerate.init_empty_weights():
        skeleton = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    got = pipelined_losses(skeleton, (3,), steps=1, weights=str(tmp_path))
    assert got == pytest.approx(expected, rel=1e-4)


def test_recompute_covers_the_vision_encoder_blocks():
    """With high-resolution images the vision encoder holds most of stage 0's
    activations, so recompute must cover its blocks too (not only the
    decoder's): 10x less kept here, against 2.3x for the decoder alone.
    Gradients are unchanged."""
    from ray_deepspeed_pipeline.hf_stage import build_hf_stage
    from ray_deepspeed_pipeline.partition import partition_parameters, recompute_blocks

    torch.manual_seed(5)
    model = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    part = partition_parameters(model, rdsp.ExplicitCuts((3,)), 2)[0]
    plain, lean = (build_hf_stage(model, 0, 3, part.parameter_names) for _ in range(2))
    recompute_blocks(lean)
    inputs, _ = make_batches()[0]
    full = {k: torch.cat(v) if isinstance(v, list) else v for k, v in inputs.items()}

    kept, grads = [], []
    for stage in (plain, lean):
        sizes = []
        with torch.autograd.graph.saved_tensors_hooks(
            lambda t, sizes=sizes: sizes.append(t.numel() * t.element_size()) or t, lambda t: t
        ):
            hidden, _ = stage(**full)
        hidden.sum().backward()
        kept.append(sum(sizes))
        grads.append({n: p.grad for n, p in stage.named_parameters()})
    assert kept[1] <= kept[0] / 5
    for name, g in grads[0].items():
        assert torch.allclose(g, grads[1][name], atol=1e-6), name


def test_vision_token_ratio_shifts_blocks_off_the_vision_stage():
    """The vision encoder's cost is its parameters times the patches it sees,
    which per row can be several times the text tokens: weighting it by that
    ratio leaves the first stage only the blocks it must keep."""
    from ray_deepspeed_pipeline.config import BalancedTransformerBlocks
    from ray_deepspeed_pipeline.partition import partition_parameters

    model = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl())
    light = partition_parameters(model, BalancedTransformerBlocks(vision_token_ratio=0.01), 2)
    heavy = partition_parameters(model, BalancedTransformerBlocks(vision_token_ratio=100.0), 2)
    assert heavy[0].block_stop == 2  # the deepstack blocks, nothing more
    assert light[0].block_stop > heavy[0].block_stop


@needs_qwen3_5
def test_right_padding_needs_no_mask():
    """Rows padded at the end: causal attention (and Qwen3.5's causal
    linear attention) already keeps real tokens off the padding, so dropping
    the padding mask leaves their loss unchanged and lets attention use its
    fastest (causal, unmasked) kernel."""
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from train_vl import drop_padding_mask

    model, _ = qwen3_5_vl_pair(seed=3)
    inputs, labels = make_batches()[0]
    inputs = {k: torch.cat(v) if isinstance(v, list) else v for k, v in inputs.items()}
    mask = torch.ones_like(inputs["input_ids"])
    mask[0, -5:] = 0  # row 0 is 5 tokens shorter
    labels = labels.masked_fill(mask == 0, -100)
    masked = model(**inputs, attention_mask=mask).logits
    unmasked = model(**drop_padding_mask({**inputs, "attention_mask": mask})).logits
    real = mask.bool()
    assert torch.allclose(masked[real], unmasked[real], atol=1e-5)
    expected = float(lm_loss(unmasked, labels))
    assert float(lm_loss(masked, labels)) == pytest.approx(expected, rel=1e-5)

    mask[1, 3] = 0  # a hole before real tokens: the mask matters there
    with pytest.raises(ValueError, match="right"):
        drop_padding_mask({**inputs, "attention_mask": mask})


def test_compile_can_cover_only_the_vision_encoder(monkeypatch):
    """A stage holding the vision encoder and decoder layers can compile just
    the encoder's blocks; the decoder layers keep their plain forward, and
    the stage computes the same thing."""
    from ray_deepspeed_pipeline.hf_stage import build_hf_stage
    from ray_deepspeed_pipeline.partition import (
        _CompiledForward,
        compile_blocks,
        partition_parameters,
    )

    real = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: real(fn, backend="eager", **kw))
    torch.manual_seed(6)
    model = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    part = partition_parameters(model, rdsp.ExplicitCuts((3,)), 2)[0]
    plain, fast = (build_hf_stage(model, 0, 3, part.parameter_names) for _ in range(2))
    compile_blocks(fast, encoder_only=True)

    vision = fast.model.model.visual.blocks
    decoder = fast.model.model.language_model.layers
    assert all(isinstance(b.forward, _CompiledForward) for b in vision)
    assert not any(isinstance(b.forward, _CompiledForward) for b in decoder)
    in_encoder = set(map(id, fast.model.model.visual.modules()))
    assert set(map(id, vision)) <= set(map(id, fast.encoder_blocks())) and all(
        id(b) in in_encoder for b in fast.encoder_blocks()
    )

    inputs, _ = make_batches()[0]
    full = {k: torch.cat(v) if isinstance(v, list) else v for k, v in inputs.items()}
    assert torch.allclose(plain(**full)[0], fast(**full)[0], atol=1e-5)


def test_one_stage_matches_unsplit_model(ray_ctx, stub_engines):
    """No pipeline at all (one stage holding the whole model, here on 2
    data-parallel ranks): the layout to compare with plain TP x DP."""
    torch.manual_seed(0)
    model = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    reference = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    reference.load_state_dict(model.state_dict())
    expected = unsplit_losses(reference, make_batches(), steps=2)
    got = pipelined_losses(model, (), steps=2, overrides=(StageOverride(stage=0, num_gpus=2),))
    assert got == pytest.approx(expected, rel=1e-4)


def ragged_batches():
    """Each microbatch padded to its own length: the step's sequences differ."""
    batches = []
    for i, (inputs, labels) in enumerate(make_batches()):
        n = SEQ - 3 * (i % 3)  # images sit at positions 2-6, kept in every length
        cut = {k: v[:, :n] if torch.is_tensor(v) and v.dim() == 2 else v for k, v in inputs.items()}
        batches.append((cut, labels[:, :n]))
    return batches


def test_microbatches_of_different_lengths_match_unsplit_model(ray_ctx, stub_engines):
    torch.manual_seed(0)
    model = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    reference = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    reference.load_state_dict(model.state_dict())
    assert len({b[1].shape[1] for b in ragged_batches()}) > 1
    expected = unsplit_losses(reference, ragged_batches(), steps=2)
    engine, _, _, _ = rdsp.initialize(
        model=model,
        config=DS,
        loss_fn=lm_loss,
        pipeline_config=rdsp.PipelineConfig(
            stages=2,
            partition=rdsp.ExplicitCuts((3,)),
            stage_overrides=(StageOverride(stage=1, num_gpus=2),),
        ),
    )
    got = [float(engine.train_batch(data_iter=iter(ragged_batches()))) for _ in range(2)]
    assert got == pytest.approx(expected, rel=1e-4)


@needs_qwen3_5
def test_colocated_vision_compile_reaches_the_plan_and_covers_the_encoder(monkeypatch):
    from ray_deepspeed_pipeline.compiler import lower
    from ray_deepspeed_pipeline.partition import _CompiledForward, compile_blocks
    from ray_deepspeed_pipeline.vision import build_vision_tower

    model = transformers.Qwen3_5ForConditionalGeneration(tiny_qwen3_5_vl())
    plan = lower(
        model,
        rdsp.PipelineConfig(
            stages=2,
            partition=rdsp.ExplicitCuts((3,)),
            colocated_vision=rdsp.ColocatedVision(compile=True),
        ),
        DS,
    )
    assert plan.colocated_vision.compile
    real = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: real(fn, backend="eager", **kw))
    tower = build_vision_tower(model, plan.colocated_vision.module)
    compile_blocks(tower)
    assert tower.local_blocks() and all(
        isinstance(b.forward, _CompiledForward) for b in tower.local_blocks()
    )
