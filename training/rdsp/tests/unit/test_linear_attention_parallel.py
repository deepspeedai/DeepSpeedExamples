# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Head-parallel gated delta-net (Qwen3.5's linear attention): two processes
stand in for two tensor-parallel ranks; each keeps half of the key and value
heads. Outputs, input gradients and every parameter gradient must equal the
unsplit layer's (its rank's slice for split parameters, the whole gradient
for parameters every rank keeps)."""

import copy
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ray_deepspeed_pipeline.linear_attention_parallel import shard_linear_attention

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5TextConfig"):
    pytest.skip("transformers without Qwen3.5", allow_module_level=True)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet  # noqa: E402

TP, HIDDEN, K_HEADS, V_HEADS, HEAD_DIM = 2, 32, 4, 8, 8


def layer(k_heads=K_HEADS, v_heads=V_HEADS):
    config = transformers.Qwen3_5TextConfig(
        hidden_size=HIDDEN,
        linear_num_key_heads=k_heads,
        linear_num_value_heads=v_heads,
        linear_key_head_dim=HEAD_DIM,
        linear_value_head_dim=HEAD_DIM,
        linear_conv_kernel_dim=4,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=16,
    )
    torch.manual_seed(0)
    full = Qwen3_5GatedDeltaNet(config, 0).float()
    with torch.no_grad():  # the norm starts at ones: make a wrong gradient visible
        full.norm.weight.normal_()
    return full


def own_rows(rank):
    """This rank's rows of the fused q/k/v projection (and conv channels):
    its key heads' slice of q and of k, then its value heads' slice of v."""
    key_dim, value_dim = K_HEADS * HEAD_DIM, V_HEADS * HEAD_DIM
    k, v = key_dim // TP, value_dim // TP
    return torch.cat(
        [
            torch.arange(rank * k, (rank + 1) * k),
            key_dim + torch.arange(rank * k, (rank + 1) * k),
            2 * key_dim + torch.arange(rank * v, (rank + 1) * v),
        ]
    )


def _reference_kernels():
    """The layer's PyTorch reference kernels: an installed GPU kernel package
    (flash-linear-attention) is otherwise picked even for CPU tensors."""
    import inspect

    from transformers.models.qwen3_5 import modeling_qwen3_5

    for name in ("torch_chunk_gated_delta_rule", "causal_conv1d_fn"):
        setattr(modeling_qwen3_5, name, inspect.unwrap(getattr(modeling_qwen3_5, name)))


def _rank(rank, init_file, out_dir):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=TP)
    _reference_kernels()
    full = layer()
    sharded = copy.deepcopy(full)
    replaced = shard_linear_attention(sharded, rank=rank, world=TP, group=lambda: dist.group.WORLD)

    x = torch.randn(2, 6, HIDDEN, generator=torch.Generator().manual_seed(1))
    target = torch.randn(2, 6, HIDDEN, generator=torch.Generator().manual_seed(2))
    x_full, x_sharded = x.clone().requires_grad_(), x.clone().requires_grad_()
    expected = full(x_full)
    (expected * target).sum().backward()
    got = sharded(x_sharded)
    (got * target).sum().backward()

    value_heads = slice(rank * V_HEADS // TP, (rank + 1) * V_HEADS // TP)
    value_cols = slice(rank * V_HEADS * HEAD_DIM // TP, (rank + 1) * V_HEADS * HEAD_DIM // TP)
    rows = own_rows(rank)

    def close(a, b):
        return a.shape == b.shape and torch.allclose(a, b, atol=1e-5)

    g = {n: p.grad for n, p in full.named_parameters()}
    torch.save(
        {
            "replaced": replaced,
            "heads": (sharded.num_k_heads, sharded.num_v_heads),
            "output": close(got, expected),
            "input_grad": close(x_sharded.grad, x_full.grad),
            "in_proj_qkv": close(sharded.in_proj_qkv.weight.grad, g["in_proj_qkv.weight"][rows]),
            "conv1d": close(sharded.conv1d.weight.grad, g["conv1d.weight"][rows]),
            "in_proj_z": close(sharded.in_proj_z.weight.grad, g["in_proj_z.weight"][value_cols]),
            "in_proj_b": close(sharded.in_proj_b.weight.grad, g["in_proj_b.weight"][value_heads]),
            "in_proj_a": close(sharded.in_proj_a.weight.grad, g["in_proj_a.weight"][value_heads]),
            "A_log": close(sharded.A_log.grad, g["A_log"][value_heads]),
            "dt_bias": close(sharded.dt_bias.grad, g["dt_bias"][value_heads]),
            "out_proj": close(sharded.out_proj.weight.grad, g["out_proj.weight"][:, value_cols]),
            "norm": close(sharded.norm.weight.grad, g["norm.weight"]),
        },
        os.path.join(out_dir, f"r{rank}.pt"),
    )
    dist.destroy_process_group()


def test_head_parallel_layer_matches_the_unsplit_layer():
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(_rank, args=(os.path.join(d, "init"), d), nprocs=TP)
        for rank in range(TP):
            r = torch.load(os.path.join(d, f"r{rank}.pt"))
            assert r.pop("replaced") == [""]  # the layer itself
            assert r.pop("heads") == (K_HEADS // TP, V_HEADS // TP)
            assert all(r.values()), {k: v for k, v in r.items() if not v}


def test_a_layer_whose_heads_do_not_split_evenly_stays_whole():
    full = layer(k_heads=3, v_heads=6)
    before = {n: p.clone() for n, p in full.named_parameters()}
    assert shard_linear_attention(full, rank=0, world=2, group=None) == []
    assert all(torch.equal(p, before[n]) for n, p in full.named_parameters())


def test_a_tp_stage_splits_its_linear_attention_and_autotp_leaves_it_alone(tmp_path):
    """Under TP the adapter splits the gated delta-net layers by heads before
    building AutoTP's rules, and drops their projections from the stage's TP
    plan; full attention and the MLP stay with AutoTP."""
    from ray_deepspeed_pipeline.deepspeed_adapter import _split_linear_attention
    from ray_deepspeed_pipeline.partition import attach_tp_plan

    config = transformers.Qwen3_5TextConfig(
        hidden_size=HIDDEN,
        linear_num_key_heads=K_HEADS,
        linear_num_value_heads=V_HEADS,
        linear_key_head_dim=HEAD_DIM,
        linear_value_head_dim=HEAD_DIM,
        linear_conv_kernel_dim=4,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        vocab_size=16,
    )
    model = transformers.Qwen3_5ForCausalLM(config)
    attach_tp_plan(model, config.base_model_tp_plan)
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'init'}", rank=0, world_size=1)
    try:
        _split_linear_attention(model, tp=2)
    finally:
        dist.destroy_process_group()
    gdn = model.model.layers[0].linear_attn
    assert (gdn.num_k_heads, gdn.num_v_heads) == (K_HEADS // 2, V_HEADS // 2)
    assert not any("linear_attn" in key for key in model._tp_plan)
    assert {"layers.*.self_attn.q_proj", "layers.*.mlp.down_proj"} <= set(model._tp_plan)
