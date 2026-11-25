#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# rdsp_zero3.sh: rdsp on Qwen3.5-35B-A3B as one stage of all 16 GPUs under DeepSpeed ZeRO-3, the
# fully sharded layout of torchtitan.sh: weights, gradients and Adam split 16 ways, one row per GPU
# per microbatch, 8 microbatches. fp32 master (4 bytes): the 2-byte master works with ZeRO 1 and 2
# only; TorchTitan's master is fp32 too. On node 1, after prep.sh, with the Ray cluster up.
# STEPS (default 50), RECOMPUTE=1 (every block), NAME (log suffix). Logs in logs/fsdp16/.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../common/two_node.sh"
export LOGS=$HERE/logs/fsdp16$LOGS_SUFFIX; mkdir -p "$LOGS"
TAG=""; RC=""; [ "$RECOMPUTE" = 1 ] && TAG=-rc RC=",recompute=1"
rdsp "rdsp-zero3$TAG${NAME:-}" --stages 1 --stage "0:gpus=16,zero=3$RC" \
  --rows 16 --microbatches 8 --master-dtype fp32
