#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# run.sh <fit|run> [CONFIG]: TorchTitan, Qwen3.5-27B TP4 x PP2 on Bridge's exported rows, on the
# H100 box after scripts/setup_box.sh, the 27B download and the export.
#   fit   3 steps (memory check)        run   50 steps (STEPS to change)
# CONFIG: qwen35_27b_cord_tp4pp2 (default, no recompute) or qwen35_27b_cord_tp4pp2_rc.
# Logs ranks 0 (stage 0) and 4 (stage 1, where the loss is) to ../logs/<step>-<CONFIG>.log.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LOGS=$HERE/../logs; mkdir -p "$LOGS"
CONFIG=${2:-qwen35_27b_cord_tp4pp2}
case "${1:-}" in
  fit) STEPS=3 ;;
  run) STEPS=${STEPS:-50} ;;
  *) sed -n 2,6p "$0"; exit 1 ;;
esac
# MODEL (default Qwen3.5-27B), EXPORT_NAME (its rows in ~/mimo, default cord_steps-$MODEL), ROWS
MODEL=${MODEL:-Qwen3.5-27B}
HF_ASSETS=$(ls -d "$HOME"/hf/hub/models--Qwen--$MODEL/snapshots/* | head -1)
cd ~/torchtitan
(unset LD_LIBRARY_PATH && . ~/tt_venv/bin/activate && \
 env PYTHONPATH="$HERE/.." HF_ASSETS="$HF_ASSETS" EXPORTED="$HOME/mimo/${EXPORT_NAME:-cord_steps-$MODEL}" \
     ROWS=${ROWS:-64} STEPS=$STEPS MODULE=cord_qwen35 CONFIG=$CONFIG NGPU=8 LOG_RANK=${LOG_RANK:-0,4} ./run_train.sh) \
  2>&1 | tee "$LOGS/$1-$CONFIG${NAME:-}.log"   # NAME: a log-name suffix
