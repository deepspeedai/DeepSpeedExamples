#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# run.sh rdsp: Qwen3.5-35B-A3B with the vision encoder on a stage of its own: K GPUs
# encode the step's images (one image row per GPU per microbatch), the language model runs as
# (16 - K) / 2 pipeline stages of 2 GPUs. K comes from run 1's measured stage times
# (scripts/layout_from_log.py vision-width ../pp8dp2/logs/rdsp.log 16), not by hand. 128 / K
# microbatches of K rows. --cuts 0,measured: the language stages re-cut by their measured times.
# On node 1, after ../scripts/prep.sh, with the Ray cluster up.
# The vision stage's encoder is recomputed (scripts/common.sh vision_first; VISION_RC=0: not).
# LANG_EP=2: each language stage splits its experts over its 2 GPUs (DeepSpeed AutoEP, NVLink
# inside the node): about half the weights and optimizer state per GPU.
# STAGE_GPUS=4: language stages of 4 GPUs instead of 2 ((16 - K) / 4 of them; K a multiple of 4),
# logs get "-s4"; with LANG_EP=4 each splits its experts over its 4 GPUs.
# K (required: 2, 4 or 8), STEPS (default 50), RECOMPUTE=1 (every block), NAME (log suffix).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../scripts/common.sh"
export LOGS=$HERE/logs$LOGS_SUFFIX; mkdir -p "$LOGS"
TAG=""; [ "$RECOMPUTE" = 1 ] && TAG=-rc
case "${1:-}" in
  rdsp)
    K=${K:?set K, the vision stage GPUs}
    SG=${STAGE_GPUS:-2}
    if [ $(( (16 - K) % SG )) != 0 ] || [ $(( K % SG )) != 0 ]; then
      echo "K=$K does not split into language stages of $SG GPUs (each microbatch's K rows over $SG)" >&2
      exit 1
    fi
    P=$(( (16 - K) / SG ))
    S=""; [ "$SG" != 2 ] && S=-s$SG
    rdsp "rdsp-k$K$S${LANG_EP:+-ep$LANG_EP}$TAG${NAME:-}" --stages $((P + 1)) --cuts 0,measured \
      $(vision_first "$K") $(stage_flags 1 $P "$SG" "${LANG_EP:+,ep=$LANG_EP}") \
      --rows "$K" --microbatches $((GBS / K)) ;;
  *) sed -n 2,8p "$0"; exit 1 ;;
esac
