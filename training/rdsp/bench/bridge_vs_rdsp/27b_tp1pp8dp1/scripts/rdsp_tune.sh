#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# rdsp_tune.sh: rdsp.sh's 27B run (TP1 x PP8 x DP1, 8 x H100) with extra knobs, everything else as
# rdsp.sh with --master-dtype remainder --cuts measured (the blog's 27B runs):
#   ROWS (default 1)   images (rows) per microbatch; 64 / ROWS microbatches a step
#   RC_BLOCKS=N        recompute only the first N decoder blocks of each stage (and stage 0's vision
#                      encoder) instead of every block; N counts at the calibration cut and moves
#                      with the measured cut, so each stage keeps the same number of blocks' activations
#   COMPILE=1          torch.compile each stage's decoder blocks; turns Liger's RMSNorm/SwiGLU off
#                      (the two crash together; the loss is rdsp's sharded loss either way)
#   LIGER=0            Liger off without compiling
#   STEPS (default 50); extra arguments go to train_vl.py
set -euo pipefail
RDSP=$(cd "$(dirname "$0")/../../../.." && pwd)
ROWS=${ROWS:-1}
RC=",recompute=1"
[ -n "${RC_BLOCKS:-}" ] && RC=",recompute_blocks=$RC_BLOCKS"
CP=""
LIGER_FLAG="--liger"
[ "${LIGER:-1}" = 0 ] && LIGER_FLAG=""
if [ "${COMPILE:-0}" = 1 ]; then CP=",compile=1"; LIGER_FLAG=""; fi
# RC_PER="n0,n1,...,n7": recompute_blocks per stage instead (early stages hold more microbatches in
# flight under 1F1B, so they can keep fewer blocks' activations than late ones)
IFS=, read -r -a PER <<< "${RC_PER:-}"
STAGES=()
for s in 0 1 2 3 4 5 6 7; do
  rc=$RC
  [ -n "${RC_PER:-}" ] && rc=",recompute_blocks=${PER[$s]}"
  STAGES+=(--stage "$s:gpus=1,tp=1,zero=1$rc$CP")
done
echo "rdsp_tune: ROWS=$ROWS RC_BLOCKS=${RC_BLOCKS:-all} RC_PER=${RC_PER:-} COMPILE=${COMPILE:-0} liger=${LIGER_FLAG:-off}"
exec python "$RDSP/train_vl.py" \
    --model Qwen/Qwen3.5-27B --dataset "${DATA:?set DATA=exported:<dir>}" \
    --stages 8 --cuts measured "${STAGES[@]}" \
    --rows "$ROWS" --microbatches $((64 / ROWS)) --seq 2048 --pad-multiple 0 \
    --steps "${STEPS:-50}" --lr 1e-5 --betas 0.9,0.95 --eps 1e-8 --weight-decay 0.0 \
    --grad-dtype bf16 --fused-adam --optimizer-group-numel 250000000 \
    --loss token-mean $LIGER_FLAG --prefetch --sharded-loss --drop-padding-mask \
    --master-dtype remainder \
    ${RDSP_EXTRA:-} "$@"
