# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""SparseGPT one-shot pruning of a single linear layer.

Adapted from the reference implementation released with
`SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot
<https://arxiv.org/abs/2301.00774>`_ (Frantar & Alistarh, ICML 2023),
https://github.com/IST-DASLab/sparsegpt (Apache-2.0), and from the RAC
reference code at https://github.com/RyanLucas3/Reasoning-Aware-Compression.

The algorithm itself is untouched by RAC: given the layer input activations
:math:`X_\\ell` it minimises :math:`\\|W_\\ell X_\\ell - \\widehat{W}_\\ell X_\\ell\\|_F^2`
subject to a sparsity constraint. What RAC changes is *which* activations end
up in :math:`X_\\ell` -- see :mod:`rac.calibration`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# The Hessian accumulation and the Cholesky solve are numerically delicate;
# TF32 silently drops ~10 bits of mantissa and makes the factorisation fail on
# ill-conditioned layers.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SparseGPT:
    """Accumulates :math:`H = 2/N \\cdot X X^\\top` for one layer, then prunes it."""

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.dev = layer.weight.device
        self.rows, self.columns = layer.weight.data.shape
        self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float32)
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor) -> None:
        """Fold one batch of layer inputs into the running Hessian.

        ``inp`` is ``(batch, seq, in_features)`` or ``(tokens, in_features)``;
        every token contributes one column of :math:`X_\\ell`.
        """
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        elif inp.dim() == 1:
            inp = inp.unsqueeze(0)
        inp = inp.t()  # (in_features, tokens)

        ntokens = inp.shape[1]
        self.H *= self.nsamples / (self.nsamples + ntokens)
        self.nsamples += ntokens
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp @ inp.t()

    @torch.no_grad()
    def prune(
        self,
        sparsity: float,
        prune_n: int = 0,
        prune_m: int = 0,
        blocksize: int = 128,
        percdamp: float = 0.01,
        max_percdamp: float = 0.5,
    ) -> float:
        """Prune the layer in place and return the reconstruction loss.

        With ``prune_n``/``prune_m`` set (e.g. 2:4) the layer is made
        semi-structured sparse and ``sparsity`` is ignored.

        The Hessian can be singular when a calibration set activates only part
        of the input space, so the damping factor is doubled and the whole
        solve retried whenever the Cholesky factorisation fails.
        """
        if self.nsamples == 0:
            raise RuntimeError("SparseGPT.prune() called before any calibration batch.")

        w_base = self.layer.weight.data.clone()
        damp_scale = percdamp

        while True:
            H = self.H.clone()
            W = w_base.clone().float()

            # Columns that never fire cannot be reconstructed; zero them out
            # and keep the Hessian invertible.
            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            W[:, dead] = 0

            diag = torch.arange(self.columns, device=self.dev)
            H[diag, diag] += damp_scale * torch.mean(torch.diag(H))

            try:
                L = torch.linalg.cholesky(H)
                Hinv = torch.cholesky_inverse(L)
                Hinv = torch.linalg.cholesky(Hinv, upper=True)
                del L, H
            except torch.linalg.LinAlgError as err:
                if damp_scale >= max_percdamp:
                    raise RuntimeError(
                        f"SparseGPT: Cholesky failed up to percdamp={damp_scale:.3f}. "
                        "The calibration set is likely too small for this layer.") from err
                damp_scale = min(damp_scale * 2, max_percdamp)
                print(f"  [sparsegpt] Cholesky failed, retrying with percdamp={damp_scale:.3f}")
                continue

            losses = torch.zeros(self.rows, device=self.dev)

            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                if prune_n == 0:
                    # Unstructured: the OBS saliency w^2 / [H^-1]_ii^2, with the
                    # threshold chosen per block of columns.
                    saliency = W1.pow(2) / torch.diag(Hinv1).reshape(1, -1).pow(2)
                    thresh = torch.sort(saliency.flatten())[0][int(saliency.numel() * sparsity)]
                    mask1 = saliency <= thresh
                else:
                    mask1 = torch.zeros_like(W1, dtype=torch.bool)

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if prune_n != 0 and i % prune_m == 0:
                        # Choose the n weights to drop inside this group of m.
                        group = (W1[:, i:i + prune_m].pow(2) /
                                 torch.diag(Hinv1)[i:i + prune_m].reshape(1, -1).pow(2))
                        mask1.scatter_(1, i + torch.topk(group, prune_n, dim=1, largest=False)[1],
                                       True)

                    q = w.clone()
                    q[mask1[:, i]] = 0
                    Q1[:, i] = q
                    losses += (w - q).pow(2) / d.pow(2)

                    # Propagate the error of column i onto the remaining
                    # columns of the block -- this is what makes SparseGPT
                    # stronger than magnitude pruning.
                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
                    Err1[:, i] = err1

                W[:, i1:i2] = Q1
                W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

            self.layer.weight.data.copy_(W.reshape_as(self.layer.weight).to(self.layer.weight.dtype))
            return (losses.sum() / 2).item()

    def free(self) -> None:
        self.H = None


def hessian_bytes(layer: nn.Linear) -> int:
    """Memory a :class:`SparseGPT` instance holds for ``layer`` (fp32 Hessian)."""
    cols = layer.weight.data.shape[1]
    return cols * cols * 4
