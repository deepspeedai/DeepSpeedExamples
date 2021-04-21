#!/bin/bash

# If you are able to install pytorch >= 1.8
# (and nccl >= 2.8.3 if you have 64 or more GPUs),
# we highly recommend you to use the NCCL-based 1-bit Lamb
# which has better performance and ease of use
# (see scripts in DeepSpeedExamples/bing_bert/1-bit_lamb/nccl
# and read the tutorial for more details:
# https://www.deepspeed.ai/tutorials/onebit-lamb/)

base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=onebit_lamb_64k_seq128_mpi_ethernet
OUTPUT_DIR=${base_dir}/bert_model_outputs

mkdir -p $OUTPUT_DIR

# NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 are used to disable infiniband. Remove it if needed.
NCCL_TREE_THRESHOLD=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 deepspeed --launcher=openmpi ${base_dir}/../../deepspeed_train.py \
--cf ${base_dir}/../../bert_large_lamb.json \
--max_seq_length 128 \
--output_dir $OUTPUT_DIR \
--deepspeed_mpi \
--deepspeed \
--deepspeed_transformer_kernel \
--print_steps 40 \
--lr_schedule "EE" \
--lr_offset 10e-4 \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/deepspeed_bsz64k_onebitlamb_config_seq128_mpi_ethernet.json \
--data_path_prefix /data/bert \
&> ${JOB_NAME}.log
