# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Head-parallel gated delta-net (Qwen3.5's linear attention) for
tensor-parallel stages.

AutoTP can split these projections only by gathering their outputs again, so
every rank would still hold each head's conv filter and run each head's
recurrence. Here each rank keeps 1/tp of the key heads and of the value heads
that read them, as Megatron does:

- the input projections, the conv filter, A_log and dt_bias keep this rank's
  heads; the heads are independent, so the recurrence needs no communication;
- out_proj keeps this rank's input columns, and the ranks' outputs are summed;
- the input gradient is summed over the ranks (each rank's heads contribute
  their share), and so is the gated norm's weight gradient (one weight shared
  by every head, which each rank sees only for its own heads).

Value heads are grouped by key head (value head j reads key head
j // (value heads / key heads)), so contiguous slices of both keep each group
on one rank.
"""

import functools

import torch
import torch.distributed as dist
import torch.nn as nn

from ray_deepspeed_pipeline.vocab_parallel import SumGradOverRanks, SumOverRanks

_LINEARS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
_SIZES = ("num_k_heads", "num_v_heads", "head_k_dim", "head_v_dim", "key_dim", "value_dim")
_PARTS = _LINEARS + _SIZES + ("conv1d", "A_log", "dt_bias", "norm")


def shard_linear_attention(module: nn.Module, rank: int, world: int, group) -> list[str]:
    """Split, in place, every gated delta-net layer in module by heads.
    group: a callable returning the tensor-parallel process group (resolved
    at call time, so it can be created after this). A layer whose key or value
    heads do not divide by world stays whole. Returns the names (relative to
    module) of the layers split."""
    split = []
    for name, layer in module.named_modules():
        if _splittable(layer, world):
            _split(layer, rank, world, group)
            split.append(name)
    return split


def _splittable(layer: nn.Module, world: int) -> bool:
    if not all(hasattr(layer, a) for a in _PARTS):
        return False
    if any(getattr(layer, a).bias is not None for a in _LINEARS + ("conv1d",)):
        return False
    if layer.in_proj_qkv.weight.is_meta:
        return False
    return layer.num_k_heads % world == 0 and layer.num_v_heads % world == 0


def _keep(param: nn.Parameter, index, dim: int = 0) -> nn.Parameter:
    return nn.Parameter(
        param.detach().index_select(dim, index).clone(), requires_grad=param.requires_grad
    )


def _split(layer: nn.Module, rank: int, world: int, group) -> None:
    k_heads, v_heads = layer.num_k_heads // world, layer.num_v_heads // world
    key_dim, value_dim = k_heads * layer.head_k_dim, v_heads * layer.head_v_dim
    device = layer.in_proj_qkv.weight.device

    def span(start, width):
        return torch.arange(start, start + width, device=device)

    keys = span(rank * key_dim, key_dim)
    values = span(rank * value_dim, value_dim)
    heads = span(rank * v_heads, v_heads)
    # the fused projection (and the conv over its output) is [q | k | v]
    qkv = torch.cat([keys, layer.key_dim + keys, 2 * layer.key_dim + values])

    for linear, rows in (
        (layer.in_proj_qkv, qkv),
        (layer.in_proj_z, values),
        (layer.in_proj_b, heads),
        (layer.in_proj_a, heads),
    ):
        linear.weight = _keep(linear.weight, rows)
        linear.out_features = len(rows)
    layer.out_proj.weight = _keep(layer.out_proj.weight, values, dim=1)
    layer.out_proj.in_features = value_dim
    conv = layer.conv1d
    conv.weight = _keep(conv.weight, qkv)
    conv.in_channels = conv.out_channels = conv.groups = len(qkv)
    layer.A_log = _keep(layer.A_log, heads)
    layer.dt_bias = _keep(layer.dt_bias, heads)

    layer.num_k_heads, layer.num_v_heads = k_heads, v_heads
    layer.key_dim, layer.value_dim = key_dim, value_dim
    if hasattr(layer, "conv_dim"):
        layer.conv_dim = len(qkv)

    layer.norm.weight.register_hook(functools.partial(_summed, group=group))
    layer.register_forward_pre_hook(
        functools.partial(_sum_input_grad, group=group), with_kwargs=True
    )
    layer.register_forward_hook(functools.partial(_sum_output, group=group))


def _summed(grad: torch.Tensor, group) -> torch.Tensor:
    grad = grad.contiguous().clone()
    dist.all_reduce(grad, group=group())
    return grad


def _sum_input_grad(module, args, kwargs, group):
    if "hidden_states" in kwargs:
        kwargs = dict(
            kwargs, hidden_states=SumGradOverRanks.apply(kwargs["hidden_states"], group())
        )
    else:
        args = (SumGradOverRanks.apply(args[0], group()), *args[1:])
    return args, kwargs


def _sum_output(module, args, output, group):
    return SumOverRanks.apply(output, group())
