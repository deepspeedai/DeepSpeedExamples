# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# This file is adapted from comm.py in Megatron-LM

import torch

from domino.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from .utils import split_tensor_along_last_dim


def reduce_tensor(tensor):
    if get_tensor_model_parallel_world_size() == 1:
        return tensor

    torch.distributed.all_reduce(tensor, group=get_tensor_model_parallel_group())
    return tensor


def split_tensor_last_dim(tensor):
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return tensor

    tensor_splits = split_tensor_along_last_dim(tensor, world_size)
    rank = get_tensor_model_parallel_rank()
    return tensor_splits[rank].contiguous()


def gather_tensor_last_dim(tensor):
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return tensor

    last_dim = tensor.dim() - 1
    rank = get_tensor_model_parallel_rank()
    gathered_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
    gathered_tensors[rank] = tensor
    torch.distributed.all_gather(gathered_tensors, tensor, group=get_tensor_model_parallel_group())
    return torch.cat(gathered_tensors, dim=last_dim).contiguous()


class CopyToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return reduce_tensor(grad_output)


class ReduceFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return reduce_tensor(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ScatterToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return split_tensor_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return gather_tensor_last_dim(grad_output)


class GatherFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return gather_tensor_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return split_tensor_last_dim(grad_output)
