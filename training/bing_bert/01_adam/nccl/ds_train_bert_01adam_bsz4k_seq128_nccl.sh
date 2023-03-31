#!/bin/bash

# This script requires pytorch >= 1.8
# (and nccl >= 2.8.3 if you have 64 or more GPUs).
# Read the tutorial for more details:
# https://www.deepspeed.ai/tutorials/zero-one-adam/

base_dir=`pwd`

JOB_NAME=01adam_bsz4k_seq128_nccl
OUTPUT_DIR=${base_dir}/bert_model_outputs

mkdir -p $OUTPUT_DIR

# NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 are used to disable infiniband. Remove it if needed.
run_cmd="NCCL_TREE_THRESHOLD=0 NCCL_DEBUG=INFO \
    deepspeed \
    ${base_dir}/../../deepspeed_train.py \
    --cf ${base_dir}/../../bert_large.json \
    --max_seq_length 128 \
    --output_dir $OUTPUT_DIR \
    --deepspeed \
    --print_steps 40 \
    --lr_schedule "LE" \
    --lr_offset 0.0 \
    --job_name $JOB_NAME \
    --deepspeed_config ${base_dir}/deepspeed_bsz4k_01adam_config_seq128_nccl.json \
    --data_path_prefix /data/bert \
    &> ${JOB_NAME}.log"

echo ${run_cmd}
eval ${run_cmd}