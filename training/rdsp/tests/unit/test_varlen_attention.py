# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""--vision-attn varlen: the vision encoder's attention through one
variable-length call per layer gives HF's SDPA results, and reads the image
lengths from the GPU once per forward instead of once per layer. PyTorch's
varlen_attn is CUDA-only, so on CPU a per-image SDPA stands in for it."""

import copy
import os
import sys

import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5VisionModel"):
    pytest.skip("transformers without Qwen3.5", allow_module_level=True)

from ray_deepspeed_pipeline import varlen_attention  # noqa: E402
from ray_deepspeed_pipeline.varlen_attention import (  # noqa: E402
    NAME,
    register_if_used,
    use_varlen_vision_attention,
)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))


def per_image_sdpa(q, k, v, cu_q, cu_k, max_q, max_k, *, scale=None, window_size=(-1, -1)):
    """varlen_attn's contract, one image at a time; (tokens, heads, head_dim)."""
    assert window_size == (-1, -1) and max_q == max_k
    bounds = [int(x) for x in cu_q]  # int(), not .tolist(): not counted as a sync below
    assert max_q == max(b - a for a, b in zip(bounds, bounds[1:]))
    out = [
        F.scaled_dot_product_attention(
            *(t[a:b].transpose(0, 1) for t in (q, k, v)), scale=scale
        ).transpose(0, 1)
        for a, b in zip(bounds, bounds[1:])
    ]
    return torch.cat(out)


@pytest.fixture
def cpu_varlen(monkeypatch):
    import torch.nn.attention.varlen as varlen

    monkeypatch.setattr(varlen, "varlen_attn", per_image_sdpa)


def model(depth=3):
    cfg = transformers.Qwen3_5VisionConfig(
        depth=depth,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=16,
        patch_size=4,
        spatial_merge_size=2,
        temporal_patch_size=1,
        num_position_embeddings=16,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    m = torch.nn.Module()
    m.visual = transformers.Qwen3_5VisionModel(cfg).float()
    return m


def images():
    grid = torch.tensor([[1, 4, 4], [1, 2, 6], [1, 6, 4]])
    pixels = torch.randn(int(grid.prod(-1).sum()), 48, generator=torch.Generator().manual_seed(1))
    return pixels, grid


def run(m):
    pixels, grid = images()
    out = m.visual(pixels, grid_thw=grid).pooler_output
    out.square().sum().backward()
    return out.detach(), [p.grad.clone() for p in m.visual.parameters()]


def test_same_features_and_gradients_as_sdpa(cpu_varlen):
    sdpa = model()
    varlen = copy.deepcopy(sdpa)
    use_varlen_vision_attention(varlen)
    assert {c._attn_implementation for c in varlen_attention._configs(varlen)} == {NAME}
    (a, ga), (b, gb) = run(sdpa), run(varlen)
    assert torch.allclose(a, b, atol=1e-6)
    for x, y in zip(ga, gb):
        assert torch.allclose(x, y, atol=1e-5)


def test_host_reads_do_not_grow_with_depth(cpu_varlen, monkeypatch):
    """SDPA reads the image lengths back three times in every layer; varlen reads only
    what the encoder reads once per forward (grid sizes, the longest image)."""
    reads = []
    for name in ("item", "tolist"):
        real = getattr(torch.Tensor, name)
        monkeypatch.setattr(
            torch.Tensor, name, lambda self, _real=real, _n=name: reads.append(_n) or _real(self)
        )
    pixels, grid = images()

    def count(depth, varlen):
        m = model(depth)
        if varlen:
            use_varlen_vision_attention(m)
        reads.clear()
        m.visual(pixels, grid_thw=grid)
        return len(reads)

    assert count(6, varlen=False) - count(2, varlen=False) == 3 * 4  # q, k and v, each layer
    assert count(6, varlen=True) == count(2, varlen=True) <= 3


def test_stage_workers_register_it_again(monkeypatch):
    from transformers import AttentionInterface

    m = model(1)
    use_varlen_vision_attention(m)
    monkeypatch.delitem(AttentionInterface._global_mapping, NAME)
    register_if_used(torch.nn.Linear(2, 2))  # names nothing: no registration
    assert NAME not in AttentionInterface._global_mapping
    register_if_used(m)
    assert AttentionInterface._global_mapping[NAME] is varlen_attention.varlen_attention_forward


def test_train_vl_flag():
    import train_vl

    assert train_vl.build_parser().parse_args([]).vision_attn == "same"
    assert train_vl.build_parser().parse_args(["--vision-attn", "varlen"]).vision_attn == "varlen"
