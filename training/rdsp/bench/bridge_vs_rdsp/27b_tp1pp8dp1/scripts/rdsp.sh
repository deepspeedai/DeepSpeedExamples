#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# rdsp, Qwen3.5-27B, TP1 x PP8 x DP1 (8 pipeline stages of one GPU) on 8x H100. 1 row(s) per microbatch (one per DP copy), 64
# microbatches per step, as Megatron runs this layout. Every row padded to 2048. Recompute every
# block (27B does not fit this layout without), bf16 gradient accumulation,
# fused AdamW, optimizer groups of 250M. Vision encoder eager. Extra arguments go to train_vl.py.
set -euo pipefail
RDSP=$(cd "$(dirname "$0")/../../../.." && pwd)
# RECOMPUTE=0: no recompute (a fit check); ZERO: ZeRO stage (default 1); PROFILER: a command
# prefix such as an nsys profile call
RC=",recompute=1"; [ "${RECOMPUTE:-1}" = 0 ] && RC=""
# RC_BLOCKS=N: recompute only each stage's first N decoder layers (and the vision encoder)
[ -n "${RC_BLOCKS:-}" ] && RC=",recompute_blocks=$RC_BLOCKS"
exec ${PROFILER:-} python "$RDSP/train_vl.py" \
    --model Qwen/Qwen3.5-27B --dataset "${DATA:?set DATA=exported:<dir>}" \
    --stages 8 --cuts balanced --stage 0:gpus=1,tp=1,zero=${ZERO:-1}$RC --stage 1:gpus=1,tp=1,zero=${ZERO:-1}$RC --stage 2:gpus=1,tp=1,zero=${ZERO:-1}$RC --stage 3:gpus=1,tp=1,zero=${ZERO:-1}$RC --stage 4:gpus=1,tp=1,zero=${ZERO:-1}$RC --stage 5:gpus=1,tp=1,zero=${ZERO:-1}$RC --stage 6:gpus=1,tp=1,zero=${ZERO:-1}$RC --stage 7:gpus=1,tp=1,zero=${ZERO:-1}$RC \
    --rows 1 --microbatches 64 --seq 2048 --pad-multiple 0 \
    --steps "${STEPS:-50}" --lr 1e-5 --betas 0.9,0.95 --eps 1e-8 --weight-decay 0.0 \
    --grad-dtype bf16 --fused-adam --optimizer-group-numel 250000000 \
    --loss token-mean --liger --prefetch --sharded-loss --drop-padding-mask \
    "$@"
