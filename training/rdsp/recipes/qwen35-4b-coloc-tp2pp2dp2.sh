#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# Qwen3.5-4B, vision encoder colocated: every GPU encodes 1/8 of the step's
# images (data parallel, no TP inside the encoder). Language model: 2 pipeline
# stages, each TP2 x DP2; rdsp places the cut (--cuts balanced picks 19
# decoder layers for stage 0 on the exported CORD-v2 samples).
#
# GPUs:     8, 80 GB class
# Measured: 5.62 s per step on 8x H100 (NVLink), DATA=exported:<dir>, before
#           --vision-compile was added (Megatron-Bridge TP2xPP2xDP2: 8.79 s)
#
# DATA: cord-v2 (default; downloads CORD-v2, images scaled to --max-pixels)
# or exported:<dir> from bench/mimo/runs.sh export (the benchmark's samples).
# Both pad each microbatch to its longest row.
# Extra arguments are passed to train_vl.py and override the flags below.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python "$ROOT/train_vl.py" \
    --model Qwen/Qwen3.5-4B --dataset "${DATA:-cord-v2}" --untie-embeddings \
    --stages 2 --cuts balanced \
    --colocated-vision --vision-compile \
    --stage 0:gpus=4,tp=2,zero=1 \
    --stage 1:gpus=4,tp=2,zero=1 \
    --rows 2 --microbatches 32 --seq 2048 --pad-multiple 128 \
    --steps 50 --lr 1e-5 --betas 0.9,0.95 --eps 1e-8 --weight-decay 0.0 \
    --grad-dtype fp32 --loss token-mean --liger \
    --prefetch --sharded-loss --drop-padding-mask --pad-per-microbatch \
    "$@"
