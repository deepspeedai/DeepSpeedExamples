#!/bin/bash
##################apply random-ltd to fine-tune ptb on GPT-base (12-layer)##############################
####see more on random-ltd: https://arxiv.org/abs/2211.11586
export CUDA_VISIBLE_DEVICES=1
mkdir -p ./output/check_base
python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port 12346 \
    run_clm_no_trainer.py \
    --random_ltd \
    --dataset_name ptb_text_only \
    --dataset_config_name penn_treebank \
    --model_name_or_path gpt2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 2  \
    --deepspeed_config config/ds_config_gpt_base_random_ltd.json \
    --deepspeed --seed 1234 --num_warmup_steps 100 \
    --output_dir ./output/check_base &> ./output/check_base/training.log

# python run_clm_no_trainer.py \
#     --random_ltd \
#     --dataset_name ptb_text_only \
#     --dataset_config_name penn_treebank \
#     --model_name_or_path gpt2 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 4 \
#     --num_train_epochs 2 \
#     --deepspeed_config config/ds_config_gpt_base_random_ltd.json \
#     --deepspeed --seed 1234\
#     --output_dir ./output/check_base