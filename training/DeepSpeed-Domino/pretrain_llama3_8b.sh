# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# This file is adapted from pretrain_llama.sh in Megatron-LM

#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,2
GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
 
CHECKPOINT_PATH=/workspace/dataset/checkpoint
TOKENIZER_PATH=/workspace/model/Llama-3.1-8B
rm -rf $CHECKPOINT_PATH/*
rm -rf ./wandb/*
VOCAB_FILE="/workspace/dataset/gpt2-vocab.json"
MERGE_FILE="/workspace/dataset/gpt2-merges.txt"
DATA_PATH="/workspace/dataset/BookCorpusDataset_text_document"
 
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PYTHONPATH=$SCRIPT_DIR:$PYTHONPATH
 
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

LLAMA_ARGS="
    --disable-bias-linear \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_PATH} \
    --normalization RMSNorm \
    --group-query-attention \
    --num-query-groups 8 \
    --no-masked-softmax-fusion \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 500000 \
    --use-rope-scaling \
    --use-rotary-position-embeddings \
    --swiglu \
    --num-layers 32  \
    --hidden-size 4096  \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32  \
    --max-position-embeddings 131072  \
    --seq-length 2048 \
    
    --micro-batch-size 4 \
    --global-batch-size 8 \
    --lr 0.00015 \
    --train-iters 80 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --tensor-model-parallel-size $WORLD_SIZE \
    --seed 3407 \
"
# llama3.1 70B
# LLAMA_ARGS="
#     --disable-bias-linear \
#     --tokenizer-type HuggingFaceTokenizer \
#     --tokenizer-model ${TOKENIZER_PATH} \
#     --transformer-impl local \
#     --normalization RMSNorm \
#     --group-query-attention \
#     --num-query-groups 8 \
#     --no-masked-softmax-fusion \
#     --attention-softmax-in-fp32 \
#     --attention-dropout 0.0 \
#     --hidden-dropout 0.0 \
#     --untie-embeddings-and-output-weights \
#     --position-embedding-type rope \
#     --rotary-percent 1.0 \
#     --rotary-base 500000 \
#     --use-rope-scaling \
#     --use-rotary-position-embeddings \
#     --swiglu \
#     --tensor-model-parallel-size 1  \
#     --pipeline-model-parallel-size 1  \
#     --num-layers 80  \
#     --hidden-size 8192  \
#     --ffn-hidden-size 28672 \
#     --num-attention-heads 64  \
#     --max-position-embeddings 131072  \
#     --seq-length 8192 \
    
#     --micro-batch-size 2 \
#     --global-batch-size 8 \
#     --lr 0.00015 \
#     --train-iters 80 \
#     --lr-decay-iters 320000 \
#     --lr-decay-style cosine \
#     --min-lr 1.0e-5 \
#     --weight-decay 1e-2 \
#     --lr-warmup-fraction .01 \
#     --clip-grad 1.0 \
#     --no-gradient-accumulation-fusion \
#     --fp16 \
#     --tensor-model-parallel-size $WORLD_SIZE \
#     --seed 3407 \
#     --causal-lm
# "

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"
 
OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 1
"
 
cmd="deepspeed --num_gpus $WORLD_SIZE \
    pretrain_llama.py \
    $LLAMA_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS 
    "
echo $cmd
eval $cmd 