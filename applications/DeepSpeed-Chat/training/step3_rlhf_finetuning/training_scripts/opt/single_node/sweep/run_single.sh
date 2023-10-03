#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ACTOR_MODEL_PATH=$1
CRITIC_MODEL_PATH=$2
ACTOR_ZERO_STAGE=$3
CRITIC_ZERO_STAGE=$4
ENABLE_HYBRID_ENGINE=$5
OFFLOAD=$6
LORA=$7
MIXED_PRECISION_LORA=$8
OUTPUT=$9
TEST=${10}
TEST_STOP_STEP=${11}

if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=2
fi

if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=2
fi

if [ "$ENABLE_HYBRID_ENGINE" == true ]; then
    ENABLE_HYBRID_ENGINE="--enable_hybrid_engine"
else
    ENABLE_HYBRID_ENGINE=""
fi

if [ "$OFFLOAD" == true ]; then
    OFFLOAD="--offload"
else
    OFFLOAD=""
fi

if [ "$LORA" == true ]; then
    ACTOR_LORA_DIM="--actor_lora_dim 128"
    ACTOR_LORA_MODULE_NAME="--actor_lora_module_name decoder.layers."
else
    ACTOR_LORA_DIM="--actor_lora_dim 0"
    ACTOR_LORA_MODULE_NAME=""
fi

if [ "$MIXED_PRECISION_LORA" == true ]; then
    MIXED_PRECISION_LORA="--enable_mixed_precision_lora"
else
    MIXED_PRECISION_LORA=""
fi

if [ "$TEST" == true ]; then
    TEST="--enable_test_mode"
    TEST_STOP_STEP="--test_stop_step ${TEST_STOP_STEP}"
else
    TEST=""
    TEST_STOP_STEP=""
fi

mkdir -p $OUTPUT

Num_Padding_at_Beginning=1 # this is model related

Actor_Lr=9.65e-6
Critic_Lr=5e-6

cmd="deepspeed --num_nodes=1 main.py \
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
   --actor_weight_decay 0 \
   --critic_weight_decay 0 \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --actor_dropout 0.0 \
   --print_answers \
   --actor_zero_stage ${ACTOR_ZERO_STAGE} \
   --critic_zero_stage ${CRITIC_ZERO_STAGE} \
   --output_dir $OUTPUT \
    $ENABLE_HYBRID_ENGINE $OFFLOAD $MIXED_PRECISION_LORA \
    $ACTOR_LORA_DIM $ACTOR_LORA_MODULE_NAME\
    $TEST $TEST_STOP_STEP"

echo "----------------------------- DS COMMAND -----------------------------"
echo $cmd

$cmd &> $OUTPUT/${OUTPUT}.log
