#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ACTOR_MODEL_PATH="AdamG012/chat-opt-1.3b-sft-deepspeed"
CRITIC_MODEL_PATH="AdamG012/chat-opt-350m-reward-deepspeed"

for z in {2..3}
do
    for he in true false
    do
        for offload in true false
        do
            for lora in true false
            do
                cmd="bash training_scripts/single_node/sweep/run_1.3b_lora_swp.sh \
                    $ACTOR_MODEL_PATH \
                    $CRITIC_MODEL_PATH \
                    ${z} \
                    ${z} \
                    ${he} \
                    ${offload} \
                    ${lora} \
                    z${z}_he_${he}_offload_${offload}_lora_${lora}"
                echo "----------------------------- CALLING SHELL SCRIPT -----------------------------"
                echo $cmd
                $cmd
                pkill -9 python
                sleep 60
                echo ""
            done
        done
    done
done
