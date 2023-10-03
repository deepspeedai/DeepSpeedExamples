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
    ACTOR_ZERO_STAGE=0
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=0
fi
mkdir -p $OUTPUT

deepspeed --num_gpus 1 main.py \
   --actor_model_name_or_path $ACTOR_MODEL_PATH --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --actor_zero_stage $ACTOR_ZERO_STAGE --critic_zero_stage $CRITIC_ZERO_STAGE \
   --num_padding_at_beginning 1 --gradient_accumulation_steps 2 \
   --deepspeed --actor_lora_dim 128 --enable_hybrid_engine --actor_gradient_checkpointing --actor_dropout 0.0 \
   --output_dir $OUTPUT &> $OUTPUT/training.log
