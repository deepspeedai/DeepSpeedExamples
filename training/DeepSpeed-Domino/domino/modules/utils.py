# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# Copied and modified from Megatron-LM

import math
import torch
from domino.arguments import get_args 

def init_method_normal(std_dev):
    def initialize(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std_dev)
    return initialize

def scaled_init_method_normal(std_dev, layer_count):
    scaled_std_dev = std_dev / math.sqrt(2.0 * layer_count)
    def initialize(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=scaled_std_dev)
    return initialize


def get_linear_layer(input_dim, output_dim, init_method):
    linear_layer = torch.nn.Linear(input_dim, output_dim)
    if get_args().perform_initialization:
        init_method(linear_layer.weight)
    with torch.no_grad():
        linear_layer.bias.zero_()
    return linear_layer

def param_is_not_shared(param):
    return not hasattr(param, 'shared') or not param.shared
