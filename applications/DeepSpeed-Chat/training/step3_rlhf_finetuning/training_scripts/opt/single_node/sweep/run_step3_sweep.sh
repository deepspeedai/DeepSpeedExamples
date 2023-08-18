#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ACTOR_MODEL_PATH="AdamG012/chat-opt-1.3b-sft-deepspeed"
CRITIC_MODEL_PATH="AdamG012/chat-opt-350m-reward-deepspeed"

# Sweep switches
RUN_GENERIC_SWEEP=true
RUN_MPL_SWEEP=true

# Kill any existing Python processes
pkill -9 python
sleep 300

# Run generic sweep w/o Mixed Precision ZeRO++
if [ "$RUN_GENERIC_SWEEP" == true ]; then
    echo "----------------------------- RUNNING GENERIC SWEEPS -----------------------------"
    echo ""
    for z in {2..3}
    do
        for he in true false
        do
            for offload in true false
            do
                for lora in true false
                do
                    mixed_precision_lora=false
                    cmd="bash training_scripts/opt/single_node/sweep/run_single.sh \
                        $ACTOR_MODEL_PATH \
                        $CRITIC_MODEL_PATH \
                        ${z} \
                        ${z} \
                        ${he} \
                        ${offload} \
                        ${lora} \
                        ${mixed_precision_lora} \
                        z${z}_he_${he}_offload_${offload}_lora_${lora}"
                    echo "----------------------------- CALLING SHELL SCRIPT -----------------------------"
                    echo $cmd
                    $cmd
                    pkill -9 python
                    sleep 300
                    echo ""
                done
            done
        done
    done
    echo ""
fi

# Run Mixed Precision ZeRO++ sweep
if [ "$RUN_MPL_SWEEP" == true ]; then
    echo "----------------------------- RUNNING MIXED PRECISION ZERO++ SWEEPS -----------------------------"
    echo ""
    for he in true false
    do
        z=3
        offload=false
        lora=true
        mixed_precision_lora=true
        cmd="bash training_scripts/opt/single_node/sweep/run_single.sh \
            $ACTOR_MODEL_PATH \
            $CRITIC_MODEL_PATH \
            ${z} \
            ${z} \
            ${he} \
            ${offload} \
            ${lora} \
            ${mixed_precision_lora} \
            z${z}_he_${he}_offload_${offload}_lora_${lora}_mpl_${mixed_precision_lora}"
        echo "----------------------------- CALLING SHELL SCRIPT -----------------------------"
        echo $cmd
        $cmd
        pkill -9 python
        sleep 300
        echo ""
    done
    echo ""
fi
