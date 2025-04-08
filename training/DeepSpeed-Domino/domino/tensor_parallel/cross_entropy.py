# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# This file is adapted from cross_entropy.py in Megatron-LM and fused_linear_cross_entropy.py in Liger-Kernel:src/liger_kernel/ops/
import torch
import triton
import triton.language as tl

from domino.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from .utils import VocabUtility

from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd
from liger_kernel.ops.utils import element_mul_kernel
from liger_kernel.ops.utils import is_hip

class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target):        
        max_logits = torch.max(logits, dim=-1)[0] # [batchsize, seq_len, 1]
        torch.distributed.all_reduce(max_logits, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group())
        
        logits = logits - max_logits.unsqueeze(dim=-1)

        partition_vocab_size = logits.size()[-1] # 25216
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        vocab_start, vocab_end = VocabUtility.vocab_range_from_per_partition_vocab_size(partition_vocab_size, rank, world_size)

        target_mask = (target < vocab_start) | (target >= vocab_end)
        adjusted_target = target.clone() - vocab_start # relative id
        adjusted_target[target_mask] = 0

        logits_2d = logits.view(-1, partition_vocab_size) # [batchsize * seq_len, vocab_size]
        adjusted_target_1d = adjusted_target.view(-1)
        batch_indices = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[batch_indices, adjusted_target_1d].clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        torch.distributed.all_reduce(predicted_logits, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group())

        exp_logits = torch.exp(logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group())
        
        loss = torch.log(sum_exp_logits) - predicted_logits # [512, 8]
        
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, adjusted_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target_mask, adjusted_target_1d = ctx.saved_tensors
        
        grad_input = softmax.view(-1, softmax.size()[-1])
        batch_indices = torch.arange(start=0, end=grad_input.size()[0], device=grad_input.device)
        softmax_update = 1.0 - target_mask.view(-1).float()
        grad_input[batch_indices, adjusted_target_1d] -= softmax_update
        grad_input = grad_input.view_as(softmax)

        grad_input.mul_(grad_output.unsqueeze(dim=-1))
        return grad_input, None

def vocab_parallel_cross_entropy(vocab_parallel_logits, target):
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target)

MAX_FUSED_SIZE = 65536 // 2

def fused_linear_cross_entropy_forward_megatron_chunked(
    _input,
    weight,
    target,
    bias=None,
    reduction="none",
):  
    device = _input.device
    BT, H = _input.shape
    V = weight.shape[0] # [V, H]

    inc_factor = triton.cdiv(V, H)  # (V + H - 1) // H
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))  # (BT + inc_factor - 1) // inc_factor
    num_chunks = triton.cdiv(BT, chunk_size)  # (BT + chunk_size - 1) // chunk_size

    grad_weight = torch.zeros_like(weight, device=device) if weight.requires_grad else None
    grad_input = torch.zeros_like(_input, device=device)
    grad_bias = torch.zeros_like(bias, device=device) if bias is not None else None
    # we use fp32 for loss accumulator
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    predicted_logits = torch.zeros(BT, dtype=torch.float32, device=device)

    # TODO: evaluate how CUDA synchronization caused by .item() affects the speed
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    vocab_start, vocab_end = VocabUtility.vocab_range_from_per_partition_vocab_size(V, rank, world_size)
    
    target_mask = (target < vocab_start) | (target >= vocab_end)
    adjusted_target = target.clone() - vocab_start # relative id
    adjusted_target[target_mask] = 0
    adjusted_target_1d = adjusted_target.view(-1)
    
    handle_grad_input_list = []
    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        # input
        _input_chunk = _input[start_idx:end_idx]  # chunk_size x H
        # when doing matmul, use the original precision
        logits_chunk = (_input_chunk @ weight.t()).float()  # chunk_size x V # since megatron has .float, I add it here.
        
        if bias is not None:
            logits_chunk = logits_chunk + bias
        # handle target
        target_chunk = adjusted_target_1d[start_idx:end_idx]  # chunk_size,
        
        # # ensure _input and target are contiguous
        # logits_chunk = logits_chunk.contiguous() # [chunk_size, vocab_size]
        # target_chunk = target_chunk.contiguous() # [chunk_size]
        
        max_logits_chunk = torch.max(logits_chunk, dim=-1)[0]
        torch.distributed.all_reduce(max_logits_chunk, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group(), async_op=False)
        logits_chunk = logits_chunk - max_logits_chunk.unsqueeze(-1)
        
        sum_exp_logits_chunk = torch.sum(torch.exp(logits_chunk), dim=-1)
        torch.distributed.all_reduce(sum_exp_logits_chunk, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group(), async_op=False)
        
        predicted_logits_chunk = logits_chunk[torch.arange(end_idx-start_idx), target_chunk]
        predicted_logits_chunk[target_mask[start_idx:end_idx]] = 0.0
        handle_predicted_logits_chunk = torch.distributed.all_reduce(predicted_logits_chunk, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group(), async_op=True)
        
        # ==> Compute gradient
        grad_logits_chunk = torch.exp(logits_chunk).div_(sum_exp_logits_chunk.unsqueeze(-1))
        grad_logits_chunk[torch.arange(end_idx-start_idx), target_chunk] -= 1.0 - target_mask[start_idx:end_idx].float() # chunk_size x V
        grad_input[start_idx:end_idx] = grad_logits_chunk.to(dtype=torch.half) @ weight # fp16 or fp32 will have different memory consumption, loss curves may be the same
        
        handle_grad_input = torch.distributed.all_reduce(grad_input[start_idx:end_idx], group=get_tensor_model_parallel_group(), async_op=True)
        handle_grad_input_list.append(handle_grad_input)

        if grad_weight is not None:
            torch.addmm(
                input=grad_weight,
                mat1=grad_logits_chunk.t().to(
                    _input_chunk.dtype
                ),  # In an autocast scenario without bias, differing logits_chunk data types will cause an addmm operation error.
                mat2=_input_chunk,
                out=grad_weight,
                alpha=1.0,
                beta=1.0, # grad_weight accumulation (beta=1.0 brings loss decrease improvement in early iterations)
            )
        if bias is not None:
            torch.add(
                input=grad_bias,
                other=grad_logits_chunk.sum(dim=0),
                out=grad_bias,
                alpha=1.0,
            )
        handle_predicted_logits_chunk.wait()
        predicted_logits[start_idx:end_idx] = predicted_logits_chunk
        loss_chunk = torch.log(sum_exp_logits_chunk) - predicted_logits_chunk
        loss_1d[start_idx:end_idx] = loss_chunk

    for handle in handle_grad_input_list:
        handle.wait()
        
    if reduction == "none":
        loss = loss_1d
    else:
        loss = torch.sum(loss_1d)
    
    return loss, None, grad_input, grad_weight, grad_bias

def fused_linear_cross_entropy_forward_megatron(
    _input,
    weight,
    target,
    bias=None,
    reduction="none",
):
    device = _input.device
    BT, H = _input.shape
    V = weight.shape[0]

    grad_weight = torch.zeros_like(weight, device=device) if weight.requires_grad else None
    grad_input = torch.zeros_like(_input, device=device)
    grad_bias = torch.zeros_like(bias, device=device) if bias is not None else None
    # we use fp32 for loss accumulator
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)

    # TODO: evaluate how CUDA synchronization caused by .item() affects the speed
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    vocab_start, vocab_end = VocabUtility.vocab_range_from_per_partition_vocab_size(V, rank, world_size)
    
    target_mask = (target < vocab_start) | (target >= vocab_end)
    adjusted_target = target.clone() - vocab_start # relative id
    adjusted_target[target_mask] = 0
    adjusted_target_1d = adjusted_target.view(-1)
    
    # input
    # when doing matmul, use the original precision
    logits = (_input @ weight.t()).float()  # chunk_size x V
    if bias is not None:
        logits = logits + bias
    
    # # ensure _input and target are contiguous
    # logits_chunk = logits_chunk.contiguous() # [chunk_size, vocab_size]
    # target_chunk = target_chunk.contiguous() # [chunk_size]
    
    max_logits = torch.max(logits, dim=-1)[0]
    torch.distributed.all_reduce(max_logits, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group(), async_op=False)
    logits = logits - max_logits.unsqueeze(-1)
    
    sum_exp_logits = torch.sum(torch.exp(logits), dim=-1)
    torch.distributed.all_reduce(sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group(), async_op=False)
    
    
    predicted_logits = logits[torch.arange(BT, device=logits.device), adjusted_target_1d]
    predicted_logits[target_mask] = 0.0
    handle_predicted_logits = torch.distributed.all_reduce(predicted_logits, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group(), async_op=True)
    
    # Compute gradient
    grad_logits = torch.exp(logits).div_(sum_exp_logits.unsqueeze(-1))
    grad_logits[torch.arange(BT, device=grad_logits.device), adjusted_target_1d] -= 1.0 - target_mask.float() # chunk_size x V
    grad_input = grad_logits.to(dtype=torch.half) @ weight
    torch.distributed.all_reduce(grad_input, group=get_tensor_model_parallel_group(), async_op=False)
    
    if grad_weight is not None:
        torch.addmm(
            input=grad_weight,
            mat1=grad_logits.t().to(
                _input.dtype
            ),  # In an autocast scenario without bias, differing logits_chunk data types will cause an addmm operation error.
            mat2=_input,
            out=grad_weight,
            alpha=1.0,
            beta=1.0,
        )
    if bias is not None:
        torch.add(
            input=grad_bias,
            other=grad_logits.sum(dim=0),
            out=grad_bias,
            alpha=1.0,
        )
    handle_predicted_logits.wait()
    loss_chunk = torch.log(sum_exp_logits) - predicted_logits
    loss_1d = loss_chunk

    if reduction == "none":
        loss = loss_1d
    else:
        loss = torch.sum(loss_1d)
    
    return loss, None, grad_input, grad_weight, grad_bias


def fused_linear_cross_entropy_backward(grad_output, grad_input, grad_weight, grad_bias):
    # If cross entropy is the last layer, grad_output is 1.0. Skip the mul to save time
    if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        # We use a Triton kernel instead of a PyTorch operation because modifying inputs in-place
        # for gradient storage and backward multiple times causes anomalies with PyTorch but not with Triton.
        BT, H = grad_input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        element_mul_kernel[(n_rows,)](
            grad_input,
            grad_input.stride(-2),
            grad_output,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )
        if grad_weight is not None:
            V, H = grad_weight.shape
            n_rows = V

            element_mul_kernel[(n_rows,)](
                grad_weight,
                grad_weight.stride(-2),
                grad_output,
                H,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )
        if grad_bias is not None:
            V = grad_bias.shape[0]
            n_rows = V

            element_mul_kernel[(n_rows,)](
                grad_bias,
                grad_bias.stride(-1),
                grad_output,
                1,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )
    return grad_input, grad_weight, grad_bias

class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        _input,
        weight,
        target,
        bias=None,
        ce_weight=None,
        ignore_index=-100,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="none",
        softcap=None,
        return_z_loss: bool = False,
    ):
        """
        Fusing the last linear layer with cross-entropy loss
            Reference: https://github.com/mgmalek/efficient_cross_entropy

        Handle the forward and backward pass of the final linear layer via cross-entropy loss by avoiding
        the materialization of the large logits tensor. Since Cross Entropy Loss is the last layer, we can
        compute the gradient at the forward pass. By doing so, we don't have to store the _input and target
        for the backward pass.

        _input: (B*T, H) where B is batch size, T is sequence length, H is hidden dimension.
        target: (B*T) where each value is in [0, V-1]
        weight: (V, H) where V is the number of classes
        bias: (V) where V is the number of classes
        ce_weight: a manual rescaling weight given to each class. If given, has to be a Tensor of size V and floating point dtype
        ignore_index: the index to ignore in the target
        label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduction: reduction to apply
        """

        loss, z_loss, grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_forward_megatron_chunked(
            _input=_input,
            weight=weight,
            target=target,
            bias=bias,
            reduction=reduction,
        )
        # downcast to dtype and store for backward
        ctx.save_for_backward(
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
            grad_bias.detach() if bias is not None else None,
        )
        ctx.return_z_loss = return_z_loss
        return loss, z_loss

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, grad_output2):
        if ctx.return_z_loss:
            del grad_output2  # z_loss is only for logging
        (grad_input, grad_weight, grad_bias) = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_backward(
            grad_output, grad_input, grad_weight, grad_bias
        )
        return (
            grad_input,
            grad_weight,
            None,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

