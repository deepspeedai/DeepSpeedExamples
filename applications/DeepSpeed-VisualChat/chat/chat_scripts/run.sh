#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
MAIN_PATH=$1

VISION_ENCODER=/blob/transformers_cache/qwen-clip
LLM=/blob/transformers_cache/Llama-2-13b-hf

export CUDA_VISIBLE_DEVICES=0  # Do multi single evaluation 
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Do multi gpu evaluation for large models (single GPU is not enough)


python chat.py \
    --lm_model_name_or_path  $LLM \
    --vision_model_name_or_path $VISION_ENCODER \
    --checkpoint_path $MAIN_PATH --enable_mmca_attention
