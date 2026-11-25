#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# bridge_pp8dp2.sh: Megatron-Bridge on Qwen3.5-35B-A3B, TP1 x PP8 x DP2 on two nodes, the layout
# of rdsp_pp8dp2.sh: no expert parallelism, 64 microbatches of 2 rows. On node 1, after prep.sh.
#   CUT="F L": layers on the first and last stage (Megatron's middle stages are equal); unset, 5
#   per stage. ../common/layout_from_log.py megatron-cut gives the cut nearest rdsp's.
#   BRIDGE_EXTRA: more Bridge overrides, e.g. dist.use_tp_pp_dp_mapping=true (each pipeline inside
#   one node, DP across the nodes: the default order, pipeline across the nodes, deadlocked twice
#   in batched send/recv).
# STEPS (default 50), RECOMPUTE=1 (every layer), NAME (log suffix). Logs in logs/pp8dp2/.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../common/two_node.sh"
export LOGS=$HERE/logs/pp8dp2$LOGS_SUFFIX; mkdir -p "$LOGS"
TAG=""; [ "$RECOMPUTE" = 1 ] && TAG=-rc
cut=""
if [ -n "${CUT:-}" ]; then
  read -r first last <<< "$CUT"
  cut="model.num_layers_in_first_pipeline_stage=$first model.num_layers_in_last_pipeline_stage=$last"
fi
bridge tp1pp8-megatron "bridge$TAG${NAME:-}" $MEG_TP1 model.expert_model_parallel_size=1 $cut \
  ${BRIDGE_EXTRA:-}
