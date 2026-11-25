#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# Qwen3.5-4B, 2 pipeline stages, each TP4 (no data parallelism). One row
# per microbatch, 64 microbatches per step, as Megatron runs this layout.
# rdsp places the cut (--cuts balanced picks 12 on the exported CORD-v2 samples).
#
# GPUs:     8, 32 GB each (peak about 26 GB per GPU)
# Measured: 13.29 s per step, 64 rows of up to 2048 tokens, 8x RTX PRO 4500
#           (PCIe), DATA=exported:<dir> (Megatron-Bridge: 15.93 s)
#
# DATA: cord-v2 (default; downloads CORD-v2, images scaled to --max-pixels)
# or exported:<dir> from bench/common/runs.sh export (the benchmark's samples).
# Both pad each microbatch to its longest row. The cut and times above were
# measured on exported data; cord-v2's smaller images change both.
# Extra arguments are passed to train_vl.py and override the flags below.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python "$ROOT/train_vl.py" \
    --model Qwen/Qwen3.5-4B --dataset "${DATA:-cord-v2}" --untie-embeddings \
    --stages 2 --cuts balanced \
    --stage 0:gpus=4,tp=4,zero=1,compile_vision=1 \
    --stage 1:gpus=4,tp=4,zero=1 \
    --rows 1 --microbatches 64 --seq 2048 --pad-multiple 128 \
    --steps 50 --lr 1e-5 --betas 0.9,0.95 --eps 1e-8 --weight-decay 0.0 \
    --grad-dtype fp32 --loss token-mean --liger \
    --prefetch --sharded-loss --drop-padding-mask --pad-per-microbatch \
    "$@"
