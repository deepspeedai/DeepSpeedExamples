# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# This file is adapted from pretrain_gpt.sh in Megatron-LM

#!/bin/bash --login

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
 
# CHECKPOINT_PATH=checkpoint
# rm -rf $CHECKPOINT_PATH/*
VOCAB_FILE="/home/code/dataset/gpt2-vocab.json"
MERGE_FILE="/home/code/dataset/gpt2-merges.txt"
DATA_PATH="/home/code/dataset/BookCorpusDataset_text_document"


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCRIPT_DIR="$SCRIPT_DIR/Megatron-LM"
export PYTHONPATH=$SCRIPT_DIR:$PYTHONPATH

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --num-layers 12 \
    --hidden-size 2560 \
    --num-attention-heads 32 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 64 \
    --global-batch-size 64 \
    --lr 0.00015 \
    --train-iters 10 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --tensor-model-parallel-size $WORLD_SIZE \
    --seed 3407
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"
 
OUTPUT_ARGS="
    --log-interval 1 \
"

cmd="deepspeed --num_gpus $WORLD_SIZE \
    pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS 
    " 

echo $cmd
eval $cmd 
