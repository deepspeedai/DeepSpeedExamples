#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# rdsp9.sh, on node 1 with the two-node Ray cluster up (bench/two_node/ray_cluster.sh): the dense
# 27B on Megatron-MIMO's 9-GPU layout, beside MIMO's own run of it: the vision encoder alone on
# 1 GPU (stage 0, --cuts 0,measured), the language model as 2 stages of TP4. Receipt rows, 128 x
# 2048 a step, 1 row a microbatch, MIMO's per-microbatch loss mean, no recompute; the memory
# settings MIMO runs with here (bf16 gradients, 2-byte master). Stages fill the driver's node
# first: the encoder and stage 1 on node 1, stage 2 on node 2. STEPS (default 10). Log:
# ../logs/rdsp-9gpu.log. Last, the median from step 4 (6 when STEPS >= 50) to the last step, as
# rdsp numbers them (from 0); the script fails if any step of that window is missing.
set -euo pipefail
export MODEL=Qwen3.5-27B EXPORT_NAME=cord_steps-Qwen3.5-27B-128 GBS=128 STEPS=${STEPS:-10}
. ~/rdsp/bench/two_node_35b/scripts/common.sh   # RDSP_COMMON, rdsp
export LOGS=~/rdsp/bench/gap_round/logs
rdsp rdsp-9gpu --stages 3 --cuts 0,measured --stage 0:gpus=1,tp=1,zero=1 \
  --stage 1:gpus=4,tp=4,zero=1 --stage 2:gpus=4,tp=4,zero=1 \
  --rows 1 --microbatches 128 --loss microbatch-mean
first=4; [ "$STEPS" -ge 50 ] && first=6
last=$(( STEPS - 1 < 50 ? STEPS - 1 : 50 ))
median=$(MEDIAN_FIRST=$first MEDIAN_MIN=$((last - first + 1)) \
  python3 "$RDSP/bench/gap_round/scripts/medians.py" rdsp "$LOGS/rdsp-9gpu.log")
[ -n "$median" ] || { echo "steps $first-$last incomplete in $LOGS/rdsp-9gpu.log" >&2; exit 1; }
echo "median over steps $first-$last: $median"
