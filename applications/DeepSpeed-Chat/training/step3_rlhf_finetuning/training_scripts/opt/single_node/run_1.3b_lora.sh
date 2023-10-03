#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team


ACTOR_ZERO_STAGE="--actor_zero_stage 2"
CRITIC_ZERO_STAGE="--critic_zero_stage 2"
ACTOR_MODEL_PATH= # Provide the ckpt path of the actor model
CRITIC_MODEL_PATH= # Provide the ckpt path of the critic model

OUTPUT="./output"

Num_Padding_at_Beginning=1 # this is model related

Actor_Lr=9.65e-6
Critic_Lr=5e-6

mkdir -p $OUTPUT

deepspeed --master_port 12346 main.py \
   --data_path Dahoas/rm-static \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 4 \
   --per_device_training_batch_size 4 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --actor_dropout 0.0 \
   ${ACTOR_ZERO_STAGE} \
   ${CRITIC_ZERO_STAGE} \
   --actor_lora_dim 128 \
   --enable_hybrid_engine \
   --actor_lora_module_name decoder.layers. \
   --output_dir $OUTPUT \
    &> $OUTPUT/training.log
