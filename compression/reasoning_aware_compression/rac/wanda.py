# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Wanda pruning of a single linear layer.

Adapted from `A Simple and Effective Pruning Approach for Large Language
Models <https://arxiv.org/abs/2306.11695>`_ (Sun et al., ICLR 2024),
https://github.com/locuslab/wanda, MIT licensed, Copyright (c) 2023 CMU Locus
Lab. The upstream license text is reproduced in
``third_party_licenses/LICENSE.wanda``.

Wanda drops the weights with the smallest :math:`|W_{ij}| \\cdot \\|X_j\\|_2`,
comparing weights *within each output row*. It needs only the per-input-channel
activation norm rather than a full Hessian, which makes it a cheap sanity check
that the RAC calibration set -- and not some SparseGPT-specific detail -- is
what drives the accuracy gains (Table 9 of the RAC paper).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Wanda:
    """Accumulates :math:`\\|X_j\\|_2^2` per input channel, then prunes."""

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.dev = layer.weight.device
        self.rows, self.columns = layer.weight.data.shape
        self.scaler_row = torch.zeros(self.columns, device=self.dev, dtype=torch.float32)
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor) -> None:
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        elif inp.dim() == 1:
            inp = inp.unsqueeze(0)
        inp = inp.t().float()  # (in_features, tokens)

        ntokens = inp.shape[1]
        self.scaler_row *= self.nsamples / (self.nsamples + ntokens)
        self.nsamples += ntokens
        self.scaler_row += inp.pow(2).sum(dim=1) / self.nsamples

    @torch.no_grad()
    def prune(self, sparsity: float, prune_n: int = 0, prune_m: int = 0, **_) -> float:
        """Prune the layer in place and return the summed dropped saliency."""
        if self.nsamples == 0:
            raise RuntimeError("Wanda.prune() called before any calibration batch.")

        W = self.layer.weight.data
        metric = W.abs().float() * torch.sqrt(self.scaler_row).reshape(1, -1)
        mask = torch.zeros_like(metric, dtype=torch.bool)

        if prune_n != 0:
            for col in range(0, self.columns, prune_m):
                group = metric[:, col:col + prune_m]
                idx = torch.topk(group, prune_n, dim=1, largest=False)[1]
                mask[:, col:col + prune_m].scatter_(1, idx, True)
        else:
            k = int(round(self.columns * sparsity))
            if k == 0:
                return 0.0
            k = min(k, self.columns - 1)
            idx = torch.topk(metric, k, dim=1, largest=False)[1]
            mask.scatter_(1, idx, True)

        dropped = metric[mask].sum().item()
        W[mask] = 0
        return dropped

    def free(self) -> None:
        self.scaler_row = None
