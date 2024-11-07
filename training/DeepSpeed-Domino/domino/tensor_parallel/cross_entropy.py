# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# This file is adapted from cross_entropy.py in Megatron-LM

import torch

from domino.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from .utils import VocabUtility


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target):
        max_logits = torch.max(logits, dim=-1)[0]
        torch.distributed.all_reduce(max_logits, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group())
        logits = logits - max_logits.unsqueeze(dim=-1)

        partition_vocab_size = logits.size()[-1]
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        vocab_start, vocab_end = VocabUtility.vocab_range_from_per_partition_vocab_size(partition_vocab_size, rank, world_size)

        target_mask = (target < vocab_start) | (target >= vocab_end)
        adjusted_target = target.clone() - vocab_start
        adjusted_target[target_mask] = 0

        logits_2d = logits.view(-1, partition_vocab_size)
        adjusted_target_1d = adjusted_target.view(-1)
        batch_indices = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[batch_indices, adjusted_target_1d].clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        torch.distributed.all_reduce(predicted_logits, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group())

        exp_logits = torch.exp(logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group())

        loss = torch.log(sum_exp_logits) - predicted_logits
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
