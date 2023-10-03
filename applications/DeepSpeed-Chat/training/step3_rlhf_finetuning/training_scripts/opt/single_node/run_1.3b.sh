#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ACTOR_MODEL_PATH=$1
CRITIC_MODEL_PATH=$2
ACTOR_ZERO_STAGE=$3
CRITIC_ZERO_STAGE=$4
OUTPUT=$5
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=2
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=2
fi

# if actor and critic model names are not provided, then use the publicly available AdamG012/chat-opt-1.3b-sft-deepspeed and AdamG012/chat-opt-350m-reward-deepspeed
if [ "$ACTOR_MODEL_PATH" == "" ]; then
    ACTOR_MODEL_PATH=AdamG012/chat-opt-1.3b-sft-deepspeed
fi
if [ "$CRITIC_MODEL_PATH" == "" ]; then
    CRITIC_MODEL_PATH=AdamG012/chat-opt-350m-reward-deepspeed
fi

mkdir -p $OUTPUT

Num_Padding_at_Beginning=1 # this is model related

Actor_Lr=9.65e-6
Critic_Lr=5e-6

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
   --actor_dropout 0.0 \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --enable_hybrid_engine \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --enable_ema \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT \
    &> $OUTPUT/training.log
