#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
mkdir -p out/cifar/
# deepspeed --include worker-0:0 --master_port 60000 main_cifar.py      \
#     --deepspeed_config config/ds_config.json  \
#     --deepspeed   --random_ltd  \
#     --dataset cifar10vit224      \
#     --seed 1234                   \
#     --printfreq 400                \
#     --arch lvits16r224             \
#     --optimizer sgd  \
#     --lr 0.0001 --seq_len 197       \
#     --scheduler constant    \
#     --epochs 14  \
#     --batchsize 32 \
#     --data_outdir check/cifar/ | tee -a check/cifar/training.log

deepspeed  --num_nodes 1 --num_gpus 1  --master_port 60000 main_cifar.py      \
    --deepspeed_config config/ds_config_cifar_random_ltd.json  \
    --deepspeed   --random_ltd  \
    --dataset cifar10vit224      \
    --seed 1234                   \
    --printfreq 400                \
    --arch vits16r224             \
    --optimizer sgd  \
    --lr 0.0001 --seq_len 197       \
    --scheduler constant    \
    --epochs 14  \
    --batchsize 128 \
    --data_outdir out/cifar/ | tee -a out/cifar/training1.log