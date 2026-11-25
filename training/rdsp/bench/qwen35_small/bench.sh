#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# bench.sh <layout>...: rdsp against Megatron on one 8-GPU node, then the table.
#
#   noncoloc   MegatronMIMO vs rdsp: vision encoder on 1 GPU, language TP2xDP2 (5 GPUs)
#   tp2        Megatron-Bridge vs rdsp: TP2xPP2xDP2
#   tp4        Megatron-Bridge vs rdsp: TP4xPP2xDP1
#   coloc      rdsp only: vision encoder on every GPU, language TP2xPP2xDP2
#   all        the four above
#
# Env: MODEL (Qwen3.5-4B), STEPS (50), ROUNDS (1), PRICE (node $/h for the cost
# column, 5.32), LOGS (~/bench/$MODEL; round r writes $LOGS/r<r>), DRY_RUN=1
# prints the plan. Converts the checkpoints and exports the samples first when
# they are missing. Each pair runs back to back, Megatron first; a layout with
# no rdsp recipe for MODEL is skipped. NSYS_DELAY + NSYS_DURATION profile every
# run with Nsight Systems (see bench/common/runs.sh). Run after bench/setup_node.sh.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RDSP=$(cd "$HERE/../.." && pwd)
export MODEL=${MODEL:-Qwen3.5-4B} STEPS=${STEPS:-50}
ROUNDS=${ROUNDS:-1}; PRICE=${PRICE:-5.32}
BASE=${LOGS:-$HOME/bench/$MODEL}
WORK=$HOME/mimo
SIZE=$(echo "${MODEL#Qwen3.5-}" | tr '[:upper:]' '[:lower:]')

usage() { sed -n 2,17p "$0"; exit 1; }
[ $# -gt 0 ] || usage
layouts=()
for arg in "$@"; do
  case $arg in
    all) layouts+=(noncoloc tp2 tp4 coloc) ;;
    noncoloc|tp2|tp4|coloc) layouts+=("$arg") ;;
    *) echo "unknown layout: $arg"; usage ;;
  esac
done

cases_of() {  # cases_of <layout>: runs.sh cases, Megatron first
  case $1 in
    noncoloc) echo noncoloc-mimo noncoloc-rdsp ;;
    tp2) echo shared-megatron shared-rdsp ;;
    tp4) echo tp4pp2-megatron tp4pp2-rdsp ;;
    coloc) echo coloc-rdsp ;;
  esac
}
recipe_of() {
  case $1 in
    noncoloc) echo "qwen35-$SIZE-vision1-tp2dp2.sh" ;;
    tp2) echo "qwen35-$SIZE-tp2pp2dp2.sh" ;;
    tp4) echo "qwen35-$SIZE-tp4pp2dp1.sh" ;;
    coloc) echo "qwen35-$SIZE-coloc-tp2pp2dp2.sh" ;;
  esac
}

failed=0
run() {  # run <log dir> <case>
  echo "run LOGS=$1 $2"
  [ -n "${DRY_RUN:-}" ] && return 0
  LOGS=$1 bash "$RDSP/bench/common/runs.sh" "$2" || { echo "FAILED: $2"; failed=1; }
}

if [ ! -d "$WORK/std/$MODEL" ] || [ ! -d "$WORK/$MODEL-mimo" ]; then
  run "$BASE" convert
fi
exported=$(ls "$WORK/cord_steps-$MODEL"/step_*.pt 2>/dev/null | wc -l)
if [ "$exported" -lt "$STEPS" ]; then
  run "$BASE" export
fi

for round in $(seq 1 "$ROUNDS"); do
  logs=$BASE/r$round
  for layout in "${layouts[@]}"; do
    recipe=$(recipe_of "$layout")
    if [ ! -f "$RDSP/recipes/$recipe" ]; then
      echo "skip $layout: no recipes/$recipe"
      continue
    fi
    for c in $(cases_of "$layout"); do run "$logs" "$c"; done
  done
  if [ -z "${DRY_RUN:-}" ]; then
    mkdir -p "$logs"
    python3 "$RDSP/bench/common/summarize.py" "$logs" --price "$PRICE" | tee "$logs/summary.txt"
  fi
done
exit $failed
