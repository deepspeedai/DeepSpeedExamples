# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Calibration-free layer-wise magnitude pruning.

The weakest baseline in the RAC paper (``MP``). It looks only at weight
magnitudes, so the calibration set -- and therefore RAC -- has no effect on it.
Useful as a floor when reporting numbers.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from .modelutils import find_linear_layers, get_decoder_layers


@torch.no_grad()
def magnitude_prune(
    model: nn.Module,
    sparsity: float,
    prune_n: int = 0,
    prune_m: int = 0,
    scope: str = "all",
    block_indices: Optional[Sequence[int]] = None,
) -> None:
    """Prune ``model`` in place, independently per layer."""
    blocks = get_decoder_layers(model)
    selected = range(len(blocks)) if block_indices is None else block_indices

    for i in selected:
        for _, layer in find_linear_layers(blocks[i], scope=scope).items():
            W = layer.weight.data
            metric = W.abs().float()

            if prune_n != 0:
                mask = torch.zeros_like(metric, dtype=torch.bool)
                for col in range(0, W.shape[1], prune_m):
                    group = metric[:, col:col + prune_m]
                    idx = torch.topk(group, prune_n, dim=1, largest=False)[1]
                    mask[:, col:col + prune_m].scatter_(1, idx, True)
            else:
                k = int(W.numel() * sparsity)
                if k == 0:
                    continue
                thresh = torch.kthvalue(metric.flatten(), k).values
                mask = metric <= thresh

            W[mask] = 0
        print(f"[magnitude] block {i} pruned")
