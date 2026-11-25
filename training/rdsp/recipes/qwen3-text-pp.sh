#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# Qwen3-0.6B (text only) on 4 GPUs: stage 0 data parallel with ZeRO-2 on 2
# GPUs, stage 1 tensor parallel on 2 GPUs. WikiText-103.
#
# GPUs:     4
# Measured: not benchmarked in this exact layout; Qwen3-0.6B against
#           Megatron-Core on H100 is in docs/BENCHMARK_RESULTS.md
#
# Extra arguments are passed to train.py and override the flags below.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python "$ROOT/train.py" \
    --model Qwen/Qwen3-0.6B \
    --stages 2 \
    --stage 0:gpus=2,zero=2 \
    --stage 1:gpus=2,tp=2 \
    --microbatches 8 --rows 4 --seq 512 \
    --steps 100 --lr 1e-5 \
    "$@"
