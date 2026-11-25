#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# Qwen3.5-2B, vision encoder alone on 1 GPU, language model TP2 x DP2 on 4.
# MegatronMIMO's non-colocated layout.
#
# GPUs:     5 (32 GB each is enough)
# Measured: 8.30 s per step, 64 rows of up to 2048 tokens, 8x RTX PRO 4500
#           (PCIe), DATA=exported:<dir> (MegatronMIMO: 9.64 s)
#
# DATA: cord-v2 (default; downloads CORD-v2, images scaled to --max-pixels)
# or exported:<dir> from bench/mimo/runs.sh export (the benchmark's samples).
# Both pad each microbatch to its longest row. The cut and times above were
# measured on exported data; cord-v2's smaller images change both.
# Extra arguments are passed to train_vl.py and override the flags below.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python "$ROOT/train_vl.py" \
    --model Qwen/Qwen3.5-2B --dataset "${DATA:-cord-v2}" \
    --stages 2 --cuts 0 \
    --stage 0:gpus=1,zero=1,compile=1 \
    --stage 1:gpus=4,tp=2,zero=1 \
    --rows 2 --microbatches 32 --seq 2048 --pad-multiple 128 \
    --steps 50 --lr 1e-5 --betas 0.9,0.95 --eps 1e-8 --weight-decay 0.0 \
    --grad-dtype fp32 --loss microbatch-mean --liger \
    --prefetch --sharded-loss --drop-padding-mask --pad-per-microbatch \
    "$@"
