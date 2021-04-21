#!/bin/bash

# If you are able to install pytorch >= 1.8
# (and nccl >= 2.8.3 if you have 64 or more GPUs),
# we highly recommend you to use the NCCL-based 1-bit Lamb
# which has better performance and ease of use
# (see scripts in DeepSpeedExamples/bing_bert/1-bit_lamb/nccl
# and read the tutorial for more details:
# https://www.deepspeed.ai/tutorials/onebit-lamb/)

base_dir=`pwd`

# Assumes job name in previous seq128 run, will resume training from epoch 150
EPOCH=150

# Where should we save checkpoints and tensorboard events?
JOB_NAME=onebit_lamb_32k_chkpt${EPOCH}_seq512_mpi_ethernet
OUTPUT_DIR=${base_dir}/bert_model_outputs

CHECKPOINT_BASE_PATH=${OUTPUT_DIR}/saved_models/onebit_lamb_64k_seq128_mpi_ethernet
CHECKPOINT_NAME=`basename ${CHECKPOINT_BASE_PATH}/epoch${EPOCH}_*`
echo "checkpoint id: $CHECKPOINT_NAME"

mkdir -p $OUTPUT_DIR

# NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 are used to disable infiniband. Remove it if needed.
NCCL_TREE_THRESHOLD=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 deepspeed --launcher=openmpi ${base_dir}/../../deepspeed_train.py \
--cf ${base_dir}/../../bert_large_lamb.json \
--max_seq_length 512 \
--output_dir $OUTPUT_DIR \
--print_steps 100 \
--deepspeed_mpi \
--deepspeed \
--deepspeed_transformer_kernel \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/deepspeed_bsz32k_onebitlamb_config_seq512_mpi_ethernet.json \
--data_path_prefix /data/bert \
--validation_data_path_prefix /data/bert \
--rewarmup \
--lr_schedule "EE" \
--attention_dropout_checkpoint \
--lr_offset 0.0 \
--load_training_checkpoint ${CHECKPOINT_BASE_PATH} \
--load_checkpoint_id ${CHECKPOINT_NAME} \
&> ${JOB_NAME}.log
