# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""AdamW whose fp32 master weights cost 2 bytes per parameter instead of 4.

An fp32 number is its bf16 rounding plus a 16-bit remainder. The bf16 model
weight already holds the top 16 bits, so the optimizer keeps only the bottom
16 (Megatron's store_param_remainders). The master stays exact fp32: the
weight is its round to nearest, as when a separate fp32 master is copied to
bf16 (ties go away from zero here, to even there).

The update itself is torch's AdamW on fp32 tensors: a chunk of the master is
rebuilt in fp32, updated by torch.optim.adamw.adamw with fp32 moments and the
gradient upcast to fp32, and split back. Only one chunk is in fp32 at a time.
"""

import torch
from torch.optim.adamw import adamw

CHUNK_NUMEL = 1 << 25  # elements rebuilt in fp32 at once: ~0.5 GB of temporaries


def split_master(master: torch.Tensor, weight: torch.Tensor, remainder: torch.Tensor) -> None:
    """Store fp32 `master` as bf16 `weight` (its round to nearest) and int16
    `remainder` (the bits the rounding dropped, signed). All three the same
    shape."""
    bits = master.view(torch.int32)
    high = bits + 0x8000  # round half up in the bit pattern: the remainder fits int16
    high >>= 16
    weight.view(torch.int16).copy_(high)
    high <<= 16
    torch.sub(bits, high, out=high)
    remainder.copy_(high)


def join_master(weight: torch.Tensor, remainder: torch.Tensor) -> torch.Tensor:
    """The fp32 master that split_master stored, as a new tensor."""
    bits = weight.view(torch.int16).to(torch.int32)
    bits <<= 16
    bits += remainder
    return bits.view(torch.float32)


class RemainderAdamW(torch.optim.Optimizer):
    """torch.optim.AdamW for bf16 parameters with an exact fp32 master: the
    parameter and a 16-bit remainder kept in the optimizer state (module doc).
    Adam's moments are fp32. Each chunk's gradient is upcast to fp32 for the
    update, as DeepSpeed upcasts gradients for an fp32 master; foreach and
    fused pick torch's kernels as in torch.optim.AdamW. A fresh state's
    remainder is 0: the master starts at the bf16 weight, as a copy would."""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        *,
        foreach=None,
        fused=None,
        chunk_numel=CHUNK_NUMEL,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                betas=tuple(betas),
                eps=eps,
                weight_decay=weight_decay,
                foreach=foreach,
                fused=fused,
            ),
        )
        self.chunk_numel = chunk_numel

    def _state(self, p, group) -> dict:
        state = self.state[p]
        if not state:
            # where torch.optim.AdamW keeps its step count
            state["step"] = (
                torch.zeros((), dtype=torch.float32, device=p.device)
                if group["fused"]
                else torch.tensor(0.0, dtype=torch.float32)
            )
            state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            state["remainder"] = torch.zeros_like(p, dtype=torch.int16)
        return state

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("RemainderAdamW takes no closure")
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dtype != torch.bfloat16:
                    raise TypeError(f"RemainderAdamW keeps bf16 parameters, not {p.dtype}")
                state = self._state(p, group)
                flat = [
                    t.view(-1)
                    for t in (p, state["remainder"], p.grad, state["exp_avg"], state["exp_avg_sq"])
                ]
                for start in range(0, p.numel(), self.chunk_numel):
                    n = min(self.chunk_numel, p.numel() - start)
                    weight, remainder, grad, exp_avg, exp_avg_sq = (
                        t.narrow(0, start, n) for t in flat
                    )
                    master = join_master(weight, remainder)
                    # each chunk takes the step count before this step; adamw adds 1
                    adamw(
                        [master],
                        [grad.float()],
                        [exp_avg],
                        [exp_avg_sq],
                        [],
                        [state["step"].clone()],
                        foreach=group["foreach"],
                        fused=group["fused"],
                        amsgrad=False,
                        beta1=beta1,
                        beta2=beta2,
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        eps=group["eps"],
                        maximize=False,
                    )
                    split_master(master, weight, remainder)
                state["step"] += 1

    def load_state_dict(self, state_dict):
        """As torch's, but keeps the saved dtypes. torch casts floating-point
        state to the parameter's dtype, which would round the fp32 moments to
        bf16 and turn the int16 remainder into floats."""
        saved = state_dict["state"]
        ids = [i for g in state_dict["param_groups"] for i in g["params"]]
        super().load_state_dict(state_dict)
        params = [p for g in self.param_groups for p in g["params"]]
        for i, p in zip(ids, params):
            for key, value in saved.get(i, {}).items():
                if key != "step":
                    self.state[p][key] = value.to(p.device)
