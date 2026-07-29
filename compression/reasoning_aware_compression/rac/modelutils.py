# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Small model-introspection helpers shared by the RAC pruners."""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

# Sub-modules we are willing to prune. Attention/MLP projections in every
# decoder-only architecture we target (Qwen2/Qwen3, Llama, Mistral) are plain
# ``nn.Linear``.
PRUNABLE_TYPES = (nn.Linear, )

MLP_KEYWORDS = ("mlp", "feed_forward", "ffn", "gate_proj", "up_proj", "down_proj")


def dtype_kwargs(dtype: torch.dtype) -> Dict[str, torch.dtype]:
    """``from_pretrained`` dtype argument, named for the installed transformers.

    The argument was renamed from ``torch_dtype`` to ``dtype`` in transformers
    4.56; the old name still works but warns, and is slated for removal.
    """
    from packaging.version import parse
    from transformers import __version__ as transformers_version

    if parse(transformers_version).release >= (4, 56):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return the stack of transformer blocks, independent of model layout.

    Handles ``model.model.layers`` (Llama/Qwen/Mistral) and
    ``model.model.decoder.layers`` (OPT).
    """
    core = getattr(model, "model", model)
    core = getattr(core, "decoder", core)
    layers = getattr(core, "layers", None)
    if layers is None:
        raise AttributeError(f"Cannot locate decoder layers on {type(model).__name__}; "
                             "RAC supports decoder-only causal LMs.")
    return layers


def get_embedding_modules(model: nn.Module) -> List[nn.Module]:
    """Modules that run *before* the first decoder block.

    These have to sit on the calibration device while we capture the inputs of
    block 0, and can go back to CPU afterwards.
    """
    core = getattr(model, "model", model)
    core = getattr(core, "decoder", core)
    names = ("embed_tokens", "embed_positions", "rotary_emb", "embed_layer_norm")
    return [getattr(core, n) for n in names if getattr(core, n, None) is not None]


def find_linear_layers(module: nn.Module, scope: str = "all") -> Dict[str, nn.Module]:
    """Collect prunable sub-modules of ``module`` keyed by qualified name.

    ``scope="mlp"`` restricts the result to feed-forward projections, which is
    the setting used for the 2:4 throughput experiments in the RAC paper (the
    attention projections are left dense).
    """
    if scope not in ("all", "mlp"):
        raise ValueError(f"Unknown scope '{scope}', expected 'all' or 'mlp'.")

    found = {}
    for name, sub in module.named_modules():
        if not isinstance(sub, PRUNABLE_TYPES):
            continue
        if scope == "mlp" and not any(k in name.lower() for k in MLP_KEYWORDS):
            continue
        found[name] = sub

    if not found:
        # Silently returning nothing here would run the whole pruning pass and
        # save a checkpoint that is still dense. The usual cause is an
        # architecture whose projections are not ``nn.Linear`` -- GPT-2 and
        # BLOOM use ``transformers.pytorch_utils.Conv1D`` -- or an MLP naming
        # convention that ``--scope mlp`` does not recognise.
        warnings.warn(
            f"No prunable layers found in {type(module).__name__} with scope='{scope}'. "
            f"RAC prunes {'/'.join(t.__name__ for t in PRUNABLE_TYPES)} sub-modules"
            + (f" whose name contains one of {MLP_KEYWORDS}" if scope == "mlp" else "")
            + "; this block will be left dense.",
            RuntimeWarning,
            stacklevel=2,
        )
    return found


def select_blocks_by_thirds(num_blocks: int, thirds: Iterable[int]) -> List[int]:
    """Indices of the decoder blocks that fall in the requested thirds.

    Thirds are 1-based: ``(1,)`` is the first third of the network, ``(1, 3)``
    the first and last thirds, ``(1, 2, 3)`` the whole model.
    """
    thirds = sorted({int(t) for t in thirds})
    for t in thirds:
        if t not in (1, 2, 3):
            raise ValueError(f"Invalid third '{t}', expected values in {{1, 2, 3}}.")

    a, b = num_blocks // 3, (2 * num_blocks) // 3
    bounds = {1: (0, a), 2: (a, b), 3: (b, num_blocks)}
    selected: List[int] = []
    for t in thirds:
        lo, hi = bounds[t]
        selected.extend(range(lo, hi))
    return selected


def prunable_layers(model: nn.Module,
                    scope: str = "all",
                    block_indices: Optional[Iterable[int]] = None) -> Dict[str, nn.Module]:
    """Prunable layers of the whole model, keyed by fully-qualified name.

    Only layers *inside decoder blocks* are returned: the embedding matrix and
    the LM head are never pruned, so counting them would misreport sparsity.
    """
    blocks = get_decoder_layers(model)
    selected = range(len(blocks)) if block_indices is None else block_indices

    wanted = set()
    for i in selected:
        wanted.update(id(layer) for layer in find_linear_layers(blocks[i], scope=scope).values())

    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, PRUNABLE_TYPES) and id(module) in wanted
    }


@torch.no_grad()
def layerwise_sparsity(model: nn.Module,
                       scope: str = "all",
                       block_indices: Optional[Iterable[int]] = None) -> List[Tuple[str, float]]:
    """Fraction of exactly-zero weights in every prunable layer."""
    report = []
    for name, layer in prunable_layers(model, scope=scope, block_indices=block_indices).items():
        w = layer.weight.data
        report.append((name, (w == 0).sum().item() / w.numel()))
    return report


@torch.no_grad()
def model_sparsity(model: nn.Module,
                   scope: str = "all",
                   block_indices: Optional[Iterable[int]] = None) -> float:
    """Fraction of exactly-zero weights across the selected prunable layers."""
    zeros, total = 0, 0
    for layer in prunable_layers(model, scope=scope, block_indices=block_indices).values():
        w = layer.weight.data
        zeros += (w == 0).sum().item()
        total += w.numel()
    return zeros / total if total else 0.0
