#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_step1_llama2_7b_lora
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

CUDA_VISIBLE_DEVICES=0 deepspeed --master_port=29600 main.py \
    --offload \
    --offload_optimizer_device nvme \
    --offload_optimizer_nvme_path /mnt/nvme0/deepspeed2 \
    --offload_optimizer_pin_memory true \
    --offload_optimizer_ratio 0.3 \
    --offload_optimizer_buffer_count 4 \
    --offload_optimizer_fast_init false \
    --offload_param_device nvme \
    --offload_param_nvme_path /mnt/nvme0/deepspeed2 \
    --offload_param_pin_memory true \
    --offload_param_buffer_size 200000000 \
    --offload_param_buffer_count 5 \
    --aio_use_gds true \
    --dtype bf16 \
    --data_path Dahoas/rm-static \
    --data_split 2,4,4 \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --max_seq_len 512 \
    --learning_rate 9.65e-6 \
    --weight_decay 0. \
    --num_train_epochs 4  \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --gradient_checkpointing \
    --zero_stage $ZERO_STAGE \
    --deepspeed \
    --lora_dim 128 \
    --lora_module_name "layers." \
    --data_output_path /tmp/data_files2 \
    --output_dir $OUTPUT \
   &> $OUTPUT/training.log
