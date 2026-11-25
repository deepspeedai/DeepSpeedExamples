#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# rdsp_pp8dp2.sh: rdsp on Qwen3.5-35B-A3B, TP1 x PP8 x DP2 on two nodes, each of the 8 pipeline
# stages on 2 GPUs of one node (stages 0-3 on node 1, 4-7 on node 2), no expert parallelism (each
# stage's 2 copies hold all its experts). 64 microbatches of 2 rows (one per copy). On node 1,
# after prep.sh, with the Ray cluster up. --cuts measured: rdsp times its stages on 3 steps and
# re-cuts by what it measured.
# Stage 0's vision encoder is recomputed (../common/two_node.sh vision_first; VISION_RC=0: not).
# LANG_EP=2: every stage splits its experts over its 2 GPUs (DeepSpeed AutoEP, NVLink inside the
# node), about half the expert weights and optimizer state per GPU; logs get "-ep2".
# STEPS (default 50), RECOMPUTE=1 (every block), NAME (log suffix). Logs in logs/pp8dp2/.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../common/two_node.sh"
export LOGS=$HERE/logs/pp8dp2$LOGS_SUFFIX; mkdir -p "$LOGS"
TAG=""; [ "$RECOMPUTE" = 1 ] && TAG=-rc
EP=${LANG_EP:+,ep=$LANG_EP}
rdsp "rdsp${LANG_EP:+-ep$LANG_EP}$TAG${NAME:-}" --stages 8 --cuts measured $(vision_first 2)$EP \
  $(stage_flags 1 7 2 "$EP") \
  --rows 2 --microbatches $((GBS / 2))
