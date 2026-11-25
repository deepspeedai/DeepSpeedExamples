#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# run.sh <titan|titan-ep8|rdsp>: Qwen3.5-35B-A3B fully sharded over all 16 GPUs:
# weights, gradients and Adam split 16 ways, each layer gathered whole just before it
# runs; one row per GPU per microbatch, 8 microbatches. On node 1, after ../scripts/prep.sh;
# rdsp needs the Ray cluster up.
#   titan      TorchTitan FSDP over 16, every expert gathered like the rest
#   titan-ep8  TorchTitan FSDP16 + expert parallelism 8 (each node's 8 GPUs split the experts)
#   titan-ep8-bf16  the same with gradients reduced in bf16, as Bridge and rdsp send theirs
#   rdsp       rdsp, one stage of 16 GPUs under DeepSpeed ZeRO-3. fp32 master (4 bytes): the
#              2-byte master works with ZeRO 1 and 2 only; TorchTitan's master is fp32 too
# STEPS (default 50; 3 for TorchTitan's fit), RECOMPUTE=1 (every block), NAME (log suffix).
# Logs in logs/ (TorchTitan: logs/<fit|run>-<config>-node<rank>.log).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../scripts/common.sh"
export LOGS=$HERE/logs$LOGS_SUFFIX; mkdir -p "$LOGS"
TAG=""; RC=""; [ "$RECOMPUTE" = 1 ] && TAG=-rc RC=",recompute=1"
SUFFIX=""; [ "$RECOMPUTE" = 1 ] && SUFFIX=_rc
STEP=run; [ "$STEPS" = 3 ] && STEP=fit
titan() {  # titan <config>
  bash "$BOTH" env LOGS="$LOGS" STEPS="$STEPS" NET="$NET" NAME="${NAME:-}" \
    NSYS_DELAY="${NSYS_DELAY:-}" NSYS_DURATION="${NSYS_DURATION:-}" \
    bash "$RDSP/bench/torchtitan/scripts/run2.sh" "$STEP" "$1$SUFFIX"
}
case "${1:-}" in
  titan)     titan qwen35_35b_a3b_cord_fsdp16 ;;
  titan-ep8) titan qwen35_35b_a3b_cord_fsdp16_ep8 ;;
  titan-ep8-bf16) titan qwen35_35b_a3b_cord_fsdp16_ep8_bf16 ;;   # gradients reduced in bf16
  rdsp)
    rdsp "rdsp-zero3$TAG${NAME:-}" --stages 1 --stage "0:gpus=16,zero=3$RC" \
      --rows 16 --microbatches 8 --master-dtype fp32 ;;
  *) sed -n 2,12p "$0"; exit 1 ;;
esac
