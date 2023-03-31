#!/bin/bash
##################apply random-ltd to fine-tune ptb on GPT-medium (24-layer)##############################
####see more on random-ltd: https://arxiv.org/abs/2211.11586
export CUDA_VISIBLE_DEVICES=2
mkdir -p ./output/check_medium
python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port 12345 \
    run_clm_no_trainer.py \
    --random_ltd \
    --dataset_name ptb_text_only \
    --dataset_config_name penn_treebank \
    --model_name_or_path gpt2-medium \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 2 \
    --deepspeed_config config/ds_config_gpt_medium_random_ltd.json \
    --deepspeed --seed 1234 --num_warmup_steps 100 \
    --output_dir ./output/check_medium &> ./output/check_medium/training.log