#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# rdsp_9b_tp4pp2.sh: rdsp, Qwen3.5-9B, 2 pipeline stages of TP4 (no data parallelism) on 8x H100.
# One row per microbatch, 64 microbatches per step, as Megatron runs this layout.
#
#   PAD=equal (default): every row padded to 2048, as Megatron-Bridge pads under PP
#   PAD=own:             each microbatch padded to its longest row (rdsp's recipe)
#   RECOMPUTE=1:         recompute every block on both stages (not used: 9B fits)
#   COMPILE_VISION=0:    no torch.compile on the vision encoder's blocks
#   STEPS (50), DATA (exported:<Bridge's samples>). Extra arguments go to train_vl.py.
set -euo pipefail
RDSP=$(cd "$(dirname "$0")/../.." && pwd)
case ${PAD:-equal} in
  equal) PAD_FLAGS="--pad-multiple 0" ;;
  own)   PAD_FLAGS="--pad-multiple 128 --pad-per-microbatch" ;;
  *) echo "PAD must be equal or own" >&2; exit 1 ;;
esac
RC=""; [ "${RECOMPUTE:-0}" = 1 ] && RC=",recompute=1"
CV=",compile_vision=1"; [ "${COMPILE_VISION:-1}" = 0 ] && CV=""
# 9B's checkpoint has its own output layer (tie_word_embeddings false): no untying needed
exec python "$RDSP/train_vl.py" \
    --model Qwen/Qwen3.5-9B --dataset "${DATA:?set DATA=exported:<dir>}" \
    --stages 2 --cuts balanced \
    --stage "0:gpus=4,tp=4,zero=1$CV$RC" \
    --stage "1:gpus=4,tp=4,zero=1$RC" \
    --rows 1 --microbatches 64 --seq 2048 $PAD_FLAGS \
    --steps "${STEPS:-50}" --lr 1e-5 --betas 0.9,0.95 --eps 1e-8 --weight-decay 0.0 \
    --grad-dtype fp32 --loss token-mean --liger \
    --prefetch --sharded-loss --drop-padding-mask \
    "$@"
