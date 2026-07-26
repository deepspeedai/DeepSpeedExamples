# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Block-by-block calibration driver for one-shot pruning.

Only one transformer block is resident on the accelerator at a time: the
calibration activations flow through the network block by block, and each
block is pruned before its (already sparse) outputs are propagated to the
next one. Peak device memory is therefore ``one block + its Hessians +
one activation slice``, which is what lets a 70B checkpoint be pruned on a
single 80GB GPU, as in the RAC paper.

This module is deliberately agnostic to *what* the calibration tokens are --
prompt-only, C4, or prompt + on-policy chain-of-thought (RAC). Swapping the
calibration set is the entire method; see :mod:`rac.calibration`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .modelutils import find_linear_layers, get_decoder_layers, get_embedding_modules
from .sparsegpt import SparseGPT, hessian_bytes
from .wanda import Wanda

# Keyword arguments a decoder block accepts that we want to replay for every
# calibration sample. Anything else (KV caches in particular) is dropped.
_REPLAYED_BLOCK_KWARGS = (
    "attention_mask",
    "position_ids",
    "position_embeddings",
    "cache_position",
)


class _StopForward(Exception):
    """Raised by the catcher to abort the forward pass after block 0's input."""


class _Catcher(nn.Module):
    """Records the hidden states and kwargs that reach the first block."""

    def __init__(self, inner: nn.Module, store: Dict):
        super().__init__()
        self.inner = inner
        self.store = store

    def forward(self, hidden_states, **kwargs):
        idx = self.store["index"]
        self.store["inps"][idx] = hidden_states.squeeze(0).to(self.store["inps"].device)
        self.store["index"] = idx + 1
        if self.store["kwargs"] is None:
            self.store["kwargs"] = {
                k: v
                for k, v in kwargs.items() if k in _REPLAYED_BLOCK_KWARGS and v is not None
            }
        raise _StopForward

    def __getattr__(self, name):
        # nn.Module.__getattr__ only runs for attributes the module does not
        # own, so this forwards lookups like ``attention_type`` that the
        # surrounding model performs on the block it wraps.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)


def get_pruner_factory(method: str):
    """Map a ``--pruning-method`` name to a per-layer pruner class."""
    factories = {"sparsegpt": SparseGPT, "wanda": Wanda}
    if method not in factories:
        raise ValueError(f"Unknown calibrated pruning method '{method}'; "
                         f"expected one of {sorted(factories)}.")
    return factories[method]


@torch.no_grad()
def capture_block_inputs(
    model: nn.Module,
    samples: Sequence[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict]:
    """Run the embedding stack and collect the inputs of decoder block 0.

    Returns ``(inps, block_kwargs)`` where ``inps`` is a CPU tensor of shape
    ``(nsamples, seqlen, hidden)``. Because every calibration sample has the
    same length, the block kwargs (causal mask, position ids, rotary
    embeddings) are identical across samples and captured once.
    """
    layers = get_decoder_layers(model)
    hidden_size = model.config.hidden_size
    dtype = next(model.parameters()).dtype
    seqlen = samples[0].numel()

    store = {
        "inps": torch.zeros((len(samples), seqlen, hidden_size), dtype=dtype, device="cpu"),
        "kwargs": None,
        "index": 0,
    }

    embeddings = get_embedding_modules(model)
    for module in embeddings:
        module.to(device)

    layers[0] = _Catcher(layers[0], store)
    try:
        for sample in samples:
            if sample.numel() != seqlen:
                raise ValueError("All calibration samples must have the same length; "
                                 f"got {sample.numel()} and {seqlen}.")
            try:
                model(sample.reshape(1, -1).to(device))
            except _StopForward:
                pass
    finally:
        layers[0] = layers[0].inner
        for module in embeddings:
            module.to("cpu")

    if store["kwargs"] is None:
        raise RuntimeError("No calibration sample reached the first decoder block.")
    return store["inps"], store["kwargs"]


@torch.no_grad()
def sequential_prune(
    model: nn.Module,
    samples: Sequence[torch.Tensor],
    method: str = "sparsegpt",
    sparsity: float = 0.5,
    prune_n: int = 0,
    prune_m: int = 0,
    scope: str = "all",
    block_indices: Optional[Sequence[int]] = None,
    device: str = "cuda",
    blocksize: int = 128,
    percdamp: float = 0.01,
    batch_size: int = 1,
) -> List[dict]:
    """Prune ``model`` in place, block by block, on ``samples``.

    Args:
        samples: calibration sequences of identical length, as produced by
            :func:`rac.calibration.build_calibration_samples`.
        method: ``"sparsegpt"`` or ``"wanda"``.
        sparsity: per-layer unstructured sparsity; ignored when ``prune_n > 0``.
        prune_n, prune_m: semi-structured ``n:m`` pattern (e.g. 2:4).
        scope: ``"all"`` prunes attention and MLP projections, ``"mlp"`` only
            the MLP ones.
        block_indices: decoder blocks to prune; ``None`` means all of them.
            Blocks outside the list still propagate activations, they are just
            left dense.
        batch_size: calibration samples per forward pass. Higher is faster but
            holds more activation memory on the device.

    Returns:
        One dict per pruned layer with its block index, name and reconstruction
        loss.
    """
    device = torch.device(device)
    pruner_cls = get_pruner_factory(method)
    layers = get_decoder_layers(model)
    selected = set(range(len(layers)) if block_indices is None else block_indices)

    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False

    inps, block_kwargs = capture_block_inputs(model, samples, device)
    outs = torch.empty_like(inps)
    stats: List[dict] = []

    for i, block in enumerate(layers):
        block.to(device)

        if i in selected:
            targets = find_linear_layers(block, scope=scope)
        else:
            targets = {}
            print(f"[rac] block {i}: left dense")

        if targets:
            pruners = {name: pruner_cls(layer) for name, layer in targets.items()}
            if method == "sparsegpt":
                budget = sum(hessian_bytes(layer) for layer in targets.values())
                print(f"[rac] block {i}: {len(targets)} layers, "
                      f"{budget / 2**30:.2f} GiB of Hessians")

            # Pre-hooks see exactly the tensor each linear layer consumes, so
            # this works regardless of how the block routes its activations.
            handles = [
                layer.register_forward_pre_hook(
                    lambda _module, args, name=name: pruners[name].add_batch(args[0]))
                for name, layer in targets.items()
            ]
            _run_block(block, inps, outs, block_kwargs, device, batch_size, store_out=False)
            for handle in handles:
                handle.remove()

            for name in targets:
                loss = pruners[name].prune(
                    sparsity,
                    prune_n=prune_n,
                    prune_m=prune_m,
                    blocksize=blocksize,
                    percdamp=percdamp,
                )
                pruners[name].free()
                stats.append({"block": i, "layer": name, "loss": loss})
                print(f"[rac] block {i} :: {name} pruned (loss {loss:.4f})")
            del pruners

        # Propagate the (now sparse) block outputs to the next block, so every
        # layer is reconstructed against the activations it will really see.
        _run_block(block, inps, outs, block_kwargs, device, batch_size, store_out=True)

        block.to("cpu")
        _empty_cache(device)
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return stats


def _run_block(
    block: nn.Module,
    inps: torch.Tensor,
    outs: torch.Tensor,
    block_kwargs: Dict,
    device: torch.device,
    batch_size: int,
    store_out: bool,
) -> None:
    """Push every calibration sample through one block."""
    for start in range(0, inps.shape[0], batch_size):
        end = min(start + batch_size, inps.shape[0])
        hidden = inps[start:end].to(device)
        out = block(hidden, **_expand_kwargs(block_kwargs, end - start))
        if isinstance(out, tuple):
            out = out[0]
        if store_out:
            outs[start:end] = out.to(outs.device, dtype=outs.dtype)
        del hidden, out


def _expand_kwargs(block_kwargs: Dict, batch: int) -> Dict:
    """Repeat the captured single-sample kwargs along the batch dimension."""
    if batch == 1:
        return dict(block_kwargs)

    expanded = {}
    for key, value in block_kwargs.items():
        if key == "position_embeddings" and isinstance(value, tuple):
            expanded[key] = tuple(v.expand(batch, *v.shape[1:]) for v in value)
        elif torch.is_tensor(value) and value.dim() >= 2 and value.shape[0] == 1:
            expanded[key] = value.expand(batch, *value.shape[1:])
        else:
            expanded[key] = value
    return expanded


def _empty_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
