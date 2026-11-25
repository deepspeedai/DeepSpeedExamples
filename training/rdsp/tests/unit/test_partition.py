# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Deterministic contiguous partitioning and name-based rebinding."""

import functools

import pytest
import torch
import torch.nn as nn

from ray_deepspeed_pipeline.config import ExplicitCuts, UniformTransformerBlocks
from ray_deepspeed_pipeline.errors import ValidationError
from ray_deepspeed_pipeline.partition import find_block_list, partition_parameters


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        return x + torch.tanh(self.fc(x))


class ToyLM(nn.Module):
    """The sequential LM layout the partitioner targets: embed, block list,
    norm, head. Shared by the other unit and CPU integration tests."""

    def __init__(self, n_blocks=6, tied=False):
        super().__init__()
        self.embed = nn.Embedding(20, 8)
        self.blocks = nn.ModuleList(Block() for _ in range(n_blocks))
        self.norm = nn.LayerNorm(8)
        self.head = nn.Linear(8, 20, bias=False)
        if tied:
            self.head.weight = self.embed.weight

    def forward(self, ids):
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


def test_find_block_list():
    name, blocks = find_block_list(ToyLM())
    assert name == "blocks" and len(blocks) == 6


def test_uniform_split_is_even_and_contiguous():
    parts = partition_parameters(ToyLM(), UniformTransformerBlocks(), 3)
    assert [(p.block_start, p.block_stop) for p in parts] == [(0, 2), (2, 4), (4, 6)]


def test_uneven_uniform_split_front_loads():
    parts = partition_parameters(ToyLM(n_blocks=7), UniformTransformerBlocks(), 3)
    assert [(p.block_start, p.block_stop) for p in parts] == [(0, 3), (3, 5), (5, 7)]


def test_explicit_cuts():
    parts = partition_parameters(ToyLM(), ExplicitCuts(cuts=(1, 5)), 3)
    assert [(p.block_start, p.block_stop) for p in parts] == [(0, 1), (1, 5), (5, 6)]


def test_pre_and_post_modules_land_on_first_and_last_stage():
    parts = partition_parameters(ToyLM(), UniformTransformerBlocks(), 3)
    assert "embed.weight" in parts[0].parameter_names
    assert "norm.weight" in parts[2].parameter_names
    assert "head.weight" in parts[2].parameter_names
    assert "blocks.2.fc.weight" in parts[1].parameter_names
    # every parameter assigned exactly once
    all_names = [n for p in parts for n in p.parameter_names]
    assert len(all_names) == len(set(all_names)) == len(list(ToyLM().named_parameters()))


def test_single_stage_owns_everything():
    (part,) = partition_parameters(ToyLM(), UniformTransformerBlocks(), 1)
    assert part.block_start == 0 and part.block_stop == 6


def test_cross_stage_tied_parameters_rejected():
    with pytest.raises(ValidationError, match="tied"):
        partition_parameters(ToyLM(tied=True), UniformTransformerBlocks(), 2)


def test_tied_parameters_fine_on_one_stage():
    (part,) = partition_parameters(ToyLM(tied=True), UniformTransformerBlocks(), 1)
    # tied pair keeps both names, both on stage 0
    assert "embed.weight" in part.parameter_names
    assert "head.weight" in part.parameter_names


def test_wrong_cut_count_rejected():
    with pytest.raises(ValidationError, match="cuts"):
        partition_parameters(ToyLM(), ExplicitCuts(cuts=(2,)), 3)


def test_out_of_range_and_unordered_cuts_rejected():
    with pytest.raises(ValidationError):
        partition_parameters(ToyLM(), ExplicitCuts(cuts=(0, 3)), 3)
    with pytest.raises(ValidationError):
        partition_parameters(ToyLM(), ExplicitCuts(cuts=(4, 2)), 3)


def test_no_block_list_rejected():
    with pytest.raises(ValidationError, match="block list"):
        partition_parameters(nn.Linear(4, 4), UniformTransformerBlocks(), 2)


# --- stage builder selection ---------------------------------------------------


def _tiny_qwen3():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(
        vocab_size=20,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return transformers.Qwen3ForCausalLM(cfg).float()


def test_plain_sequential_model_gets_the_generic_builder():
    from ray_deepspeed_pipeline.partition import build_stage_module, select_stage_builder

    assert select_stage_builder(ToyLM()) is build_stage_module


def test_hf_causal_lm_gets_the_causal_lm_builder():
    from ray_deepspeed_pipeline.partition import build_causal_lm_stage, select_stage_builder

    assert select_stage_builder(_tiny_qwen3()) is build_causal_lm_stage


def test_rotary_model_without_the_causal_lm_layout_is_rejected():
    """A rotary embedding next to the blocks means the layers need positions;
    without the rest of the llama-style layout neither builder is right."""
    from ray_deepspeed_pipeline.partition import select_stage_builder

    class Rotary(nn.Module):
        def forward(self, x, position_ids):
            return x

    class Body(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(Block() for _ in range(4))
            self.rotary_emb = Rotary()

    class RotaryOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Body()

    with pytest.raises(ValidationError, match="rotary embedding"):
        select_stage_builder(RotaryOnly())


def _saved_bytes(stage, x):
    """Bytes autograd keeps for backward during one forward of `stage`."""
    saved = []

    def pack(t):
        saved.append(t.numel() * t.element_size())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = stage(x)
    return sum(saved), out


def test_recompute_keeps_less_for_backward_and_same_gradients():
    from ray_deepspeed_pipeline.partition import build_stage_module, recompute_blocks

    torch.manual_seed(0)
    model = ToyLM(n_blocks=6)
    middle = partition_parameters(model, ExplicitCuts((1, 5)), 3)[1]  # blocks only
    plain = build_stage_module(model, 1, 5, middle.parameter_names)
    lean = build_stage_module(model, 1, 5, middle.parameter_names)
    recompute_blocks(lean)
    hidden = torch.randn(4, 16, 8)

    kept_plain, out_plain = _saved_bytes(plain, hidden)
    kept_lean, out_lean = _saved_bytes(lean, hidden)
    out_plain.sum().backward()
    out_lean.sum().backward()

    # each block keeps its input instead of its input, weight and output
    assert kept_lean <= kept_plain / 2
    assert torch.equal(out_lean, out_plain)
    for (name, a), (_, b) in zip(plain.named_parameters(), lean.named_parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-6), name


def test_compiled_blocks_match_and_ship_to_another_process(monkeypatch):
    """Each block's forward is compiled on its first call, in the process that
    runs it: the wrapped stage pickles (Ray ships it) without the compiled
    code, and computes what the plain stage does."""
    import pickle

    from ray_deepspeed_pipeline.partition import build_stage_module, compile_blocks

    compiled = []
    real = torch.compile

    def counting(fn, **kw):
        compiled.append(kw)
        return real(fn, backend="eager", **kw)  # no C++ toolchain needed

    monkeypatch.setattr(torch, "compile", counting)
    torch.manual_seed(0)
    model = ToyLM(n_blocks=4)
    middle = partition_parameters(model, ExplicitCuts((1, 3)), 3)[1]
    plain = build_stage_module(model, 1, 3, middle.parameter_names)
    fast = build_stage_module(model, 1, 3, middle.parameter_names)
    compile_blocks(fast)
    fast = pickle.loads(pickle.dumps(fast))
    assert compiled == []
    hidden = torch.randn(4, 16, 8)

    out_plain, out_fast = plain(hidden), fast(hidden)
    out_plain.sum().backward()
    out_fast.sum().backward()
    fast(hidden)  # compiled once per block, then reused

    assert len(compiled) == 2 and all(kw.get("dynamic") for kw in compiled)
    assert torch.allclose(out_fast, out_plain, atol=1e-6)
    for (name, a), (_, b) in zip(plain.named_parameters(), fast.named_parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-6), name


def test_balanced_cuts_give_the_heavy_head_stage_fewer_blocks():
    """ToyLM: blocks cost 72 parameters each, the norm and head 176, the
    embedding nothing (a lookup). Three stages: uniform 2|2|2 costs
    144|144|320; the best split is 3|2|1 at 216|144|248 (ties go to the
    earlier stages)."""
    from ray_deepspeed_pipeline.config import BalancedTransformerBlocks

    parts = partition_parameters(ToyLM(n_blocks=6), BalancedTransformerBlocks(), 3)
    assert [(p.block_start, p.block_stop) for p in parts] == [(0, 3), (3, 5), (5, 6)]


def test_balanced_cuts_spread_the_slack_evenly():
    """Eight ToyLM blocks on four stages: 3|3|1|1 and 3|2|2|1 both cap the
    slowest stage at 248, but the first leaves stage 2 nearly idle (72)."""
    from ray_deepspeed_pipeline.config import BalancedTransformerBlocks

    parts = partition_parameters(ToyLM(n_blocks=8), BalancedTransformerBlocks(), 4)
    assert [(p.block_start, p.block_stop) for p in parts] == [(0, 3), (3, 5), (5, 7), (7, 8)]


def test_vision_token_ratio_counts_real_tokens_and_the_encoders_attention():
    """Encoder cost per text token for BalancedTransformerBlocks: the step's
    patches over the text tokens actually computed (each microbatch at its
    own length, not the configured maximum), weighted up for attention over
    each image's patches, which grows with the patches per image."""
    from types import SimpleNamespace

    from ray_deepspeed_pipeline.partition import vision_token_ratio

    config = SimpleNamespace(vision_config=SimpleNamespace(hidden_size=8, intermediate_size=16))
    layer = 4 * 8 * 8 + 2 * 8 * 16  # 512 parameters per encoder layer
    sample = [
        (
            {
                "input_ids": torch.zeros(2, 10, dtype=torch.long),
                "image_grid_thw": [torch.tensor([[1, 4, 4]]), torch.tensor([[1, 2, 4]])],
            },
            None,
        ),
        (
            {
                "input_ids": torch.zeros(2, 6, dtype=torch.long),
                "image_grid_thw": [torch.tensor([[1, 4, 4]]), torch.zeros(0, 3, dtype=torch.long)],
            },
            None,
        ),
    ]
    images = [16, 8, 16]  # patches per image; the empty grid is a row without an image
    weighted = sum(p * (1 + 2 * p * 8 / layer) for p in images)
    assert vision_token_ratio(config, sample) == pytest.approx(weighted / (2 * 10 + 2 * 6))


def test_recompute_first_blocks_only():
    """recompute_blocks(stage, n): the first n of the stage's blocks keep
    only their input, the rest keep everything; gradients unchanged."""
    from ray_deepspeed_pipeline.partition import build_stage_module, recompute_blocks

    torch.manual_seed(0)
    model = ToyLM(n_blocks=6)
    middle = partition_parameters(model, ExplicitCuts((1, 5)), 3)[1]  # blocks 1-4
    stages = {n: build_stage_module(model, 1, 5, middle.parameter_names) for n in (0, 2, 4, None)}
    for n, stage in stages.items():
        if n != 0:
            recompute_blocks(stage, n)
    wrapped = [
        sum(isinstance(b.forward, functools.partial) for b in s.local_blocks())
        for s in stages.values()
    ]
    assert wrapped == [0, 2, 4, 4]
    assert [isinstance(b.forward, functools.partial) for b in stages[2].local_blocks()] == [
        True,
        True,
        False,
        False,
    ]

    hidden = torch.randn(4, 16, 8)
    kept = {}
    for n, stage in stages.items():
        kept[n], out = _saved_bytes(stage, hidden)
        out.sum().backward()
    assert kept[4] < kept[2] < kept[0] and kept[4] == kept[None]
    for name, a in stages[0].named_parameters():
        b = dict(stages[2].named_parameters())[name]
        assert torch.allclose(a.grad, b.grad, atol=1e-6), name


def test_recompute_first_blocks_takes_the_encoder_whole():
    from ray_deepspeed_pipeline.partition import recompute_blocks

    class Stage(nn.Module):  # an HF stage holding a vision encoder
        def __init__(self):
            super().__init__()
            self.decoder = nn.ModuleList(nn.Linear(2, 2) for _ in range(3))
            self.encoder = nn.ModuleList(nn.Linear(2, 2) for _ in range(2))

        def local_blocks(self):
            return list(self.decoder) + self.encoder_blocks()

        def encoder_blocks(self):
            return list(self.encoder)

    stage = Stage()
    recompute_blocks(stage, 1)
    assert [isinstance(b.forward, functools.partial) for b in stage.local_blocks()] == [
        True,
        False,
        False,
        True,
        True,
    ]


def test_per_head_norms_get_their_tp_gradient_sum_when_the_config_plan_omits_them():
    # Qwen3-VL-MoE's own TP plan leaves out q_norm/k_norm: under tp>1 their
    # gradients would stay per rank and the TP replicas drift apart
    transformers = pytest.importorskip("transformers")
    from ray_deepspeed_pipeline.hf_stage import build_hf_stage

    torch.manual_seed(0)
    cfg = transformers.Qwen3VLMoeConfig(
        text_config=dict(
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=16,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_experts=4,
            num_experts_per_tok=2,
            vocab_size=64,
            tie_word_embeddings=False,
        ),
        vision_config=dict(
            depth=2,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            out_hidden_size=32,
            deepstack_visual_indexes=[],
        ),
        tie_word_embeddings=False,
    )
    model = transformers.Qwen3VLMoeForConditionalGeneration(cfg)
    assert "layers.*.self_attn.q_norm" not in cfg.text_config.base_model_tp_plan
    parts = partition_parameters(model, UniformTransformerBlocks(), 2)
    stage = build_hf_stage(
        model, parts[1].block_start, parts[1].block_stop, parts[1].parameter_names, 1
    )
    assert set(stage._rdsp_tp_grad_allreduce) == {
        "layers.*.self_attn.q_norm",
        "layers.*.self_attn.k_norm",
    }
    assert stage._tp_plan["layers.*.self_attn.q_proj"] == "colwise"
