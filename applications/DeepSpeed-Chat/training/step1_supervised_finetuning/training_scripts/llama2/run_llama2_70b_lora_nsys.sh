#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

source /home/ckr/miniconda3/etc/profile.d/conda.sh
conda activate ds

export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./exp/gds_70b_1gpu_raid1M_ioblock1M
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

# 设置 Hugging Face token（从环境变量读取，如果未设置则使用默认值）
# 建议在 ~/.bashrc 或 ~/.zshrc 中设置：export HF_TOKEN="your_token_here"
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable is not set. Please set it before running this script."
    echo "You can set it by: export HF_TOKEN='your_token_here'"
fi
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

# 可选：如果网络连接有问题，可以设置镜像（取消下面的注释）
# export HF_ENDPOINT="https://hf-mirror.com"

# nsys profile输出路径
NSYS_OUTPUT=$OUTPUT/nsys_profile
# Build nsys profile command
# 注意：--capture-range=cudaProfilerApi 需要代码中调用 cudaProfilerStart/Stop
# 如果代码中没有这些调用，应该移除 --capture-range 参数，让 nsys 捕获整个程序运行过程
# --delay 可以跳过模型加载阶段，只捕获训练过程
NSYS_CMD="/usr/local/cuda/bin/nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=cpu \
    --output=$NSYS_OUTPUT \
    --force-overwrite=true \
    --stats=true \
    --cuda-memory-usage=true \
    --trace-fork-before-exec=true \
    --capture-range=cudaProfilerApi \
    --gpu-metrics-devices=0 \
    --delay=240 \
    --duration=500"
    # --delay=120 \
    #  \
    #  \
CUDA_VISIBLE_DEVICES=0 $NSYS_CMD \
    deepspeed  --master_port=29600 main.py \
    --aio_block_size=1048576 \
    --offload \
    --offload_optimizer_device nvme \
    --offload_optimizer_nvme_path /mnt/raid0 \
    --offload_optimizer_pin_memory true \
    --offload_optimizer_ratio 0.3 \
    --offload_optimizer_buffer_count 8 \
    --offload_optimizer_fast_init false \
    --offload_param_device nvme \
    --offload_param_nvme_path /mnt/raid0 \
    --offload_param_pin_memory true \
    --offload_param_buffer_size 360349696 \
    --offload_param_buffer_count 32 \
    --offload_param_max_in_cpu 0 \
    --aio_use_gds true \
    --dtype bf16 \
    --data_path Dahoas/rm-static \
    --data_split 2,4,4 \
    --model_name_or_path /home/ckr/LoRA/models/70B \
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
    --data_output_path ./data \
    --output_dir $OUTPUT \
   &> $OUTPUT/training.log

