#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# Qwen3.5-4B, vision encoder alone on 1 GPU, language model TP4
# on 4.
# MegatronMIMO non-colocated layout (vision and language on disjoint GPUs).
#
# GPUs:     5, 80 GB class
# Measured: not yet
#
# DATA: cord-v2 (default; downloads CORD-v2, images scaled to --max-pixels)
# or exported:<dir> from bench/mimo/runs.sh export (the benchmark's samples).
# Both pad each microbatch to its longest row.
# Extra arguments are passed to train_vl.py and override the flags below.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python "$ROOT/train_vl.py" \
    --model Qwen/Qwen3.5-4B --dataset "${DATA:-cord-v2}" \
    --stages 2 --cuts 0 \
    --stage 0:gpus=1,zero=1,compile=1 \
    --stage 1:gpus=4,tp=4,zero=1 \
    --rows 1 --microbatches 64 \
    --seq 2048 --pad-multiple 128 \
    --steps 50 --lr 1e-5 --betas 0.9,0.95 --eps 1e-8 --weight-decay 0.0 \
    --grad-dtype fp32 --loss microbatch-mean --liger \
    --prefetch --sharded-loss --drop-padding-mask --pad-per-microbatch \
    "$@"
