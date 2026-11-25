#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# torchtitan.sh <fsdp16|fsdp16-ep8|fsdp16-ep8-bf16>: TorchTitan on Qwen3.5-35B-A3B, fully sharded
# over all 16 GPUs: weights, gradients and Adam split 16 ways, each layer gathered whole just
# before it runs; one row per GPU per microbatch, 8 microbatches. On node 1, after prep.sh.
#   fsdp16           FSDP over 16, every expert gathered like the rest
#   fsdp16-ep8       FSDP16 + expert parallelism 8 (each node's 8 GPUs split the experts)
#   fsdp16-ep8-bf16  the same with gradients reduced in bf16, as Bridge and rdsp send theirs
# STEPS (default 50; 3 for a fit check), RECOMPUTE=1 (every block), NAME (log suffix).
# Logs in logs/fsdp16/<fit|run>-<config>-node<rank>.log.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../common/two_node.sh"
export LOGS=$HERE/logs/fsdp16$LOGS_SUFFIX; mkdir -p "$LOGS"
SUFFIX=""; [ "$RECOMPUTE" = 1 ] && SUFFIX=_rc
STEP=run; [ "$STEPS" = 3 ] && STEP=fit
titan() {  # titan <config>
  bash "$BOTH" env LOGS="$LOGS" STEPS="$STEPS" NET="$NET" NAME="${NAME:-}" \
    NSYS_DELAY="${NSYS_DELAY:-}" NSYS_DURATION="${NSYS_DURATION:-}" \
    bash "$RDSP/bench/torchtitan/run_two_node.sh" "$STEP" "$1$SUFFIX"
}
case "${1:-}" in
  fsdp16)          titan qwen35_35b_a3b_cord_fsdp16 ;;
  fsdp16-ep8)      titan qwen35_35b_a3b_cord_fsdp16_ep8 ;;
  fsdp16-ep8-bf16) titan qwen35_35b_a3b_cord_fsdp16_ep8_bf16 ;;   # gradients reduced in bf16
  *) sed -n 4,11p "$0"; exit 1 ;;
esac
