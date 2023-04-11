#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

deepspeed --num_gpus 1 main.py --model_name_or_path facebook/opt-350m \
   --num_padding_at_beginning 1 --gradient_accumulation_steps 2 --zero_stage $ZERO_STAGE \
   --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.log
