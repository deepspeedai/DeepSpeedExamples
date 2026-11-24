#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# run.sh <rdsp|bridge>: Qwen3.5-35B-A3B, TP1 x PP8 x DP2 on two nodes, each of
# the 8 pipeline stages on 2 GPUs of one node (stages 0-3 on node 1, 4-7 on node 2), no expert
# parallelism (each stage's 2 copies hold all its experts). 64 microbatches of 2 rows (one per
# copy). On node 1, after ../scripts/prep.sh; rdsp needs the Ray cluster up.
#   rdsp     --cuts measured: rdsp times its stages on 3 steps and re-cuts by what it measured
#   bridge   CUT="F L": layers on the first and last stage (Megatron's middle stages are
#            equal); unset, 5 per stage. scripts/layout_from_log.py megatron-cut gives the cut
#            nearest rdsp's. BRIDGE_EXTRA: more Bridge overrides, e.g.
#            dist.use_tp_pp_dp_mapping=true (each pipeline inside one node, DP across the nodes:
#            the default order, pipeline across the nodes, deadlocked twice in batched send/recv).
# Stage 0's vision encoder is recomputed (scripts/common.sh vision_first; VISION_RC=0: not).
# LANG_EP=2: every stage splits its experts over its 2 GPUs (DeepSpeed AutoEP, NVLink inside the
# node), about half the expert weights and optimizer state per GPU; logs get "-ep2".
# STEPS (default 50), RECOMPUTE=1 (every block), NAME (log suffix). Logs in logs/.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../scripts/common.sh"
export LOGS=$HERE/logs$LOGS_SUFFIX; mkdir -p "$LOGS"
TAG=""; [ "$RECOMPUTE" = 1 ] && TAG=-rc
case "${1:-}" in
  rdsp)
    EP=${LANG_EP:+,ep=$LANG_EP}
    rdsp "rdsp${LANG_EP:+-ep$LANG_EP}$TAG${NAME:-}" --stages 8 --cuts measured $(vision_first 2)$EP \
      $(stage_flags 1 7 2 "$EP") \
      --rows 2 --microbatches $((GBS / 2)) ;;
  bridge)
    cut=""
    if [ -n "${CUT:-}" ]; then
      read -r first last <<< "$CUT"
      cut="model.num_layers_in_first_pipeline_stage=$first model.num_layers_in_last_pipeline_stage=$last"
    fi
    bridge tp1pp8-megatron "bridge$TAG${NAME:-}" $MEG_TP1 model.expert_model_parallel_size=1 $cut \
      ${BRIDGE_EXTRA:-} ;;
  *) sed -n 2,11p "$0"; exit 1 ;;
esac
