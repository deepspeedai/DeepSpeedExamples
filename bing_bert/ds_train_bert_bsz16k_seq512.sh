#!/bin/bash

num_nodes=$DLWS_NUM_WORKER
num_gpus=$DLWS_NUM_GPU_PER_WORKER
batch_size=16384

# This micro batch size assumes 32GB V100 GPUs
micro_batch=8

base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=lamb_16k_chkpt150_seq512 
OUTPUT_DIR=${base_dir}/bert_model_outputs

# Assumes job name in previous seq128 run, will resume training from epoch 150
CHECKPOINT_BASE_PATH=${OUTPUT_DIR}/saved_models/lamb_16k_seq128
CHECKPOINT_EPOCH150_NAME=`basename ${CHECKPOINT_BASE_PATH}/epoch150_*`
echo "checkpoint id: $CHECKPOINT_EPOCH150_NAME"

mkdir -p $OUTPUT_DIR
total_gpus=$(( $num_gpus * $num_nodes ))
total_micro_batch=$(( ${total_gpus} * ${micro_batch} ))
gas=$(( ${batch_size} / ${total_micro_batch} ))
train_batch_size=$(( ${micro_batch} * ${gas} ))
echo "gradient accumulation steps: ${gas}, micro batch size: $micro_batch"

deepspeed.pt ${base_dir}/deepspeed_train.py \
--cf ${base_dir}/bert_large_lamb_seq512.json \
--train_batch_size ${batch_per_gpu} \
--max_seq_length 512 \
--output_dir $OUTPUT_DIR \
--gradient_accumulation_steps ${gas} \
--max_grad_norm 1.0 \
--fp16 \
--deepspeed \
--loss_scale 0 \
--delay_allreduce \
--max_steps 48860 \
--print_steps 1 \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/deepspeed_bsz16K_lamb_config_seq512.json \
--rewarmup \
--load_training_checkpoint ${CHECKPOINT_BASE_PATH} \
--load_checkpoint_id ${CHECKPOINT_EPOCH150_NAME}

