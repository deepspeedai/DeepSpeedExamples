# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

from typing import Optional
import torch


class AddDropoutFuseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, bias1, residual1, input2, bias2, residual2, prob, training):
        if bias1 is not None and bias2 is not None:
            output1, mask1, output2, mask2 = torch._C._nn.native_add_dropout_add_fuse(
                input1, bias1, residual1, input2, bias2, residual2, prob, training
            )
        else:
            output1, mask1, output2, mask2 = torch._C._nn.native_add_dropout_fuse(
                input1, residual1, input2, residual2, prob, training
            )
        scale = 1.0 / (1.0 - prob)
        ctx.save_for_backward(mask1, mask2)
        ctx.scale = scale
        ctx.with_bias = bias1 is not None and bias2 is not None
        return output1, output2

    @staticmethod
    def backward(ctx, grad_output1, grad_output2):
        (mask1, mask2) = ctx.saved_tensors
        scale = ctx.scale
        with_bias = ctx.with_bias
        if with_bias:
            grad_input1, grad_bias1, grad_residual1, grad_input2, grad_bias2, grad_residual2 = (
                torch._C._nn.native_add_dropout_add_fuse_2(grad_output1, mask1, grad_output2, mask2, scale)
            )
        else:
            grad_input1, grad_residual1, grad_input2, grad_residual2 = (
                torch._C._nn.native_add_dropout_fuse_2(grad_output1, mask1, grad_output2, mask2, scale)
            )
            grad_bias1 = None
            grad_bias2 = None
        return grad_input1, grad_bias1, grad_residual1, grad_input2, grad_bias2, grad_residual2, None, None


class AddDropoutFuse(torch.nn.Module):
    def __init__(self):
        super(AddDropoutFuse, self).__init__()

    def forward(self, input1, bias1, residual1, input2, bias2, residual2, prob, training):
        return AddDropoutFuseFunction.apply(input1, bias1, residual1, input2, bias2, residual2, prob, training)


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Optional[Tensor], Tensor, float, bool) -> Tensor
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


@torch.jit.script
def bias_dropout_add_fused_train(x: torch.Tensor,
                                 bias: Optional[torch.Tensor],
                                 residual: torch.Tensor,
                                 prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x: torch.Tensor,
                                     bias: Optional[torch.Tensor],
                                     residual: torch.Tensor,
                                     prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, False)


def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * freqs.cos()) + (_rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim=-1)