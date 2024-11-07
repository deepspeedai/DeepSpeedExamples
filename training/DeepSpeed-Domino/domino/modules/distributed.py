# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# This file is adapted from distributed.py in Megatron-LM

import math
import torch
import domino.parallel_state as mpu


class FlattenMemory:

    def __init__(self, numel, dtype):
        self.numel = numel
        data_parallel_world_size = mpu.get_data_parallel_world_size()
        self.numel_padded = data_parallel_world_size * \
                int(math.ceil(numel / data_parallel_world_size))
        self.dtype = dtype
        self.data = torch.zeros(self.numel_padded,
                                dtype=self.dtype,
                                device=torch.cuda.current_device(),
                                requires_grad=False)


    def get(self, shape, start_index):
        end_index = start_index + shape.numel()
        assert end_index <= self.numel
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor


class DistributedDataParallel(torch.nn.Module):

    def __init__(self, module,
                 accumulate_allreduce_grads_in_fp32,
                 use_contiguous_buffers):

        super(DistributedDataParallel, self).__init__()

        self.module = module
        self.accumulate_allreduce_grads_in_fp32 \
            = accumulate_allreduce_grads_in_fp32
        self.use_contiguous_buffers = use_contiguous_buffers

        if self.accumulate_allreduce_grads_in_fp32:
            assert self.use_contiguous_buffers

        if not self.use_contiguous_buffers:
            self._grad_buffers = None
            self._grad_buffer_param_index_map = None
            return
        
        self._grad_buffers = {}
        self._grad_buffer_param_index_map = {}

        def _get_buffer_type(param):
            return torch.float if \
                self.accumulate_allreduce_grads_in_fp32 else param.dtype

        type_num_elements = {}
        for param in self.module.parameters():
            if param.requires_grad:
                dtype = _get_buffer_type(param)
                type_num_elements[dtype] = type_num_elements.get(dtype, 0) \
                                            + param.data.nelement()

        # Allocate the memory.
        for dtype, num_elements in type_num_elements.items():
            self._grad_buffers[dtype] = FlattenMemory(num_elements, dtype)

        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                dtype = _get_buffer_type(param)
                type_num_elements[dtype] -= param.data.nelement()
                param.main_grad = self._grad_buffers[dtype].get(
                    param.data.shape, type_num_elements[dtype])
                if dtype not in self._grad_buffer_param_index_map:
                    self._grad_buffer_param_index_map[dtype] = {}
                self._grad_buffer_param_index_map[dtype][param] = (
                    type_num_elements[dtype],
                    type_num_elements[dtype] + param.data.nelement(),
                )
                # Backward hook.
                param_tmp = param.expand_as(param)
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_param_hook(param))
                self.grad_accs.append(grad_acc)


    def _make_param_hook(self, param):
        """Create the all-reduce hook for backprop."""
        # Hook used for back-prop.
        def param_hook(*unused):
            # Add the gradient to the buffer.
            if param.grad is not None:
                # The gradient function of linear layers is fused with GEMMs
                param.main_grad.add_(param.grad.data)
                # Now we can deallocate grad memory.
                param.grad = None
        return param_hook


    def zero_grad_buffer(self):
        assert self._grad_buffers is not None, 'buffers are not initialized.'
        for _, buffer_ in self._grad_buffers.items():
            buffer_.data.zero_()


    def broadcast_params(self):
        for param in self.module.parameters():
            torch.distributed.broadcast(param.data,
                                        src=mpu.get_data_parallel_src_rank(),
                                        group=mpu.get_data_parallel_group())


    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
