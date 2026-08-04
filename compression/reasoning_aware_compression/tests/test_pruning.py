# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""CPU-only tests for the pruners and the block-by-block driver.

A tiny randomly-initialised Qwen2 is built from a config, so the tests are
offline and run in a few seconds. Run with ``pytest`` from
``compression/reasoning_aware_compression``.
"""

import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rac import (  # noqa: E402
    get_decoder_layers,
    magnitude_prune,
    model_sparsity,
    prunable_layers,
    select_blocks_by_thirds,
    sequential_prune,
)

SEQLEN = 32
NSAMPLES = 4


@pytest.fixture
def model():
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        torch_dtype="float32",
    )
    tiny = Qwen2ForCausalLM(config)
    tiny.eval()
    return tiny


@pytest.fixture
def samples():
    generator = torch.Generator().manual_seed(1)
    return [torch.randint(0, 256, (SEQLEN, ), generator=generator) for _ in range(NSAMPLES)]


def layer_sparsities(model, scope="all"):
    return {
        name: (layer.weight.data == 0).float().mean().item()
        for name, layer in prunable_layers(model, scope=scope).items()
    }


@pytest.mark.parametrize("method", ["sparsegpt", "wanda"])
def test_sequential_prune_hits_target_sparsity(model, samples, method):
    stats = sequential_prune(model, samples, method=method, sparsity=0.5, device="cpu")

    assert len(stats) == 3 * 7  # 3 blocks x (q,k,v,o,gate,up,down)
    for name, sparsity in layer_sparsities(model).items():
        assert sparsity == pytest.approx(0.5, abs=0.05), name


def test_sequential_prune_semi_structured_2_of_4(model, samples):
    sequential_prune(model, samples, method="sparsegpt", prune_n=2, prune_m=4, device="cpu")

    for name, layer in prunable_layers(model).items():
        W = layer.weight.data
        groups = W.reshape(W.shape[0], -1, 4)
        zeros_per_group = (groups == 0).sum(dim=2)
        assert torch.all(zeros_per_group == 2), f"{name} is not 2:4 sparse"


def test_sequential_prune_respects_block_selection(model, samples):
    # Prune only the first third of a 3-block model, i.e. block 0.
    block_indices = select_blocks_by_thirds(3, [1])
    assert block_indices == [0]

    sequential_prune(model, samples, sparsity=0.5, block_indices=block_indices, device="cpu")

    sparsities = layer_sparsities(model)
    assert all(v == pytest.approx(0.5, abs=0.05) for k, v in sparsities.items()
               if k.startswith("model.layers.0."))
    assert all(v == 0.0 for k, v in sparsities.items() if not k.startswith("model.layers.0."))


def test_sequential_prune_mlp_scope_leaves_attention_dense(model, samples):
    sequential_prune(model, samples, sparsity=0.5, scope="mlp", device="cpu")

    for name, sparsity in layer_sparsities(model, scope="all").items():
        if "mlp" in name:
            assert sparsity == pytest.approx(0.5, abs=0.05), name
        else:
            assert sparsity == 0.0, name


def test_sequential_prune_batched_matches_unbatched(model, samples):
    import copy

    batched = copy.deepcopy(model)
    sequential_prune(model, samples, sparsity=0.5, device="cpu", batch_size=1)
    sequential_prune(batched, samples, sparsity=0.5, device="cpu", batch_size=NSAMPLES)

    for (name, a), (_, b) in zip(prunable_layers(model).items(),
                                 prunable_layers(batched).items()):
        assert torch.allclose(a.weight, b.weight, atol=1e-4), name


def test_sequential_prune_rejects_ragged_samples(model):
    ragged = [torch.zeros(SEQLEN, dtype=torch.long), torch.zeros(SEQLEN + 1, dtype=torch.long)]
    with pytest.raises(ValueError, match="same length"):
        sequential_prune(model, ragged, sparsity=0.5, device="cpu")


def test_sequential_prune_rejects_unknown_method(model, samples):
    with pytest.raises(ValueError, match="Unknown calibrated pruning method"):
        sequential_prune(model, samples, method="alps", device="cpu")


@pytest.mark.parametrize("failing_kwargs, match", [
    ({"scope": "bogus"}, "Unknown scope"),
    ({"method": "alps"}, "Unknown calibrated pruning method"),
])
def test_sequential_prune_restores_use_cache_on_failure(model, samples, failing_kwargs, match):
    """A failed pruning run must leave the config as it found it."""
    model.config.use_cache = True

    with pytest.raises(ValueError, match=match):
        sequential_prune(model, samples, sparsity=0.5, device="cpu", **failing_kwargs)

    assert model.config.use_cache is True


def test_sequential_prune_restores_use_cache_on_success(model, samples):
    model.config.use_cache = True
    sequential_prune(model, samples, sparsity=0.5, device="cpu")
    assert model.config.use_cache is True


def test_find_linear_layers_warns_when_nothing_is_prunable(model):
    """Silently returning {} would save a checkpoint that is still dense."""
    import warnings

    from transformers.pytorch_utils import Conv1D

    from rac import find_linear_layers

    class Conv1DBlock(torch.nn.Module):
        """Stand-in for a GPT-2/BLOOM block, whose projections are not nn.Linear."""

        def __init__(self):
            super().__init__()
            self.attn = Conv1D(8, 8)

    with pytest.warns(RuntimeWarning, match="No prunable layers found"):
        assert find_linear_layers(Conv1DBlock()) == {}

    # A supported block finds its layers and warns about nothing.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(find_linear_layers(get_decoder_layers(model)[0])) == 7


def test_magnitude_prune(model):
    magnitude_prune(model, sparsity=0.5)
    assert model_sparsity(model) == pytest.approx(0.5, abs=0.02)


def test_pruned_model_still_runs(model, samples):
    sequential_prune(model, samples, sparsity=0.5, device="cpu")
    with torch.no_grad():
        logits = model(samples[0].reshape(1, -1)).logits
    assert logits.shape == (1, SEQLEN, 256)
    assert torch.isfinite(logits).all()


def test_calibration_data_changes_the_mask(model, samples):
    """The premise of RAC: which tokens you calibrate on changes what is cut."""
    import copy

    other_generator = torch.Generator().manual_seed(99)
    other_samples = [
        torch.randint(0, 256, (SEQLEN, ), generator=other_generator) for _ in range(NSAMPLES)
    ]

    a, b = model, copy.deepcopy(model)
    sequential_prune(a, samples, sparsity=0.5, device="cpu")
    sequential_prune(b, other_samples, sparsity=0.5, device="cpu")

    masks_differ = any(
        not torch.equal(la.weight == 0, lb.weight == 0)
        for la, lb in zip(prunable_layers(a).values(), prunable_layers(b).values()))
    assert masks_differ


def test_sparsegpt_handles_dead_input_channels():
    """A channel that never fires has a zero Hessian diagonal; it must not crash."""
    from rac.sparsegpt import SparseGPT

    torch.manual_seed(0)
    layer = torch.nn.Linear(8, 4, bias=False)
    activations = torch.randn(16, 8)
    activations[:, 3] = 0.0

    pruner = SparseGPT(layer)
    pruner.add_batch(activations)
    pruner.prune(0.5)

    assert torch.all(layer.weight.data[:, 3] == 0)
    assert torch.isfinite(layer.weight.data).all()


@pytest.mark.parametrize("cls_name", ["SparseGPT", "Wanda"])
def test_pruner_requires_calibration(cls_name):
    import rac

    pruner = getattr(rac, cls_name)(torch.nn.Linear(8, 4, bias=False))
    with pytest.raises(RuntimeError, match="before any calibration batch"):
        pruner.prune(0.5)


def test_select_blocks_by_thirds():
    assert select_blocks_by_thirds(9, [1]) == [0, 1, 2]
    assert select_blocks_by_thirds(9, [1, 3]) == [0, 1, 2, 6, 7, 8]
    assert select_blocks_by_thirds(9, [1, 2, 3]) == list(range(9))
    assert select_blocks_by_thirds(10, [2]) == [3, 4, 5]

    with pytest.raises(ValueError, match="Invalid third"):
        select_blocks_by_thirds(9, [4])
