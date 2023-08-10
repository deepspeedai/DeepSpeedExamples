#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
for z in {2..3}
do
    for offload in true false
    do
        for lora in true false
        do
            cmd="bash training_scripts/opt/single_node/sweep/run_single.sh \
                ${z} \
                ${offload} \
                ${lora} \
                z${z}_offload_${offload}_lora_${lora}"
            echo "----------------------------- CALLING SHELL SCRIPT -----------------------------"
            echo $cmd
            $cmd
            pkill -9 python
            sleep 60
            echo ""
        done
    done
done
