#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# pp8.sh <step>: 27B TP1 x PP8 x DP1 (8 pipeline stages of one GPU) on the H100 box, after
# bench/setup_node.sh and the 27B checkpoint + export (bench/qwen35_27b/prep.sh makes both).
# Logs in logs/pp8/.
#
#   bridge   Megatron-Bridge, 50 steps
#   rdsp     rdsp, 50 steps
#   timers   Megatron-Bridge, 15 steps with its own per-rank timers (logger.timing_log_level=2,
#            every rank printed) -> logs/pp8/timers-bridge<NAME>.log
#   pyspy    Megatron-Bridge, 15 steps untraced; PYSPY_SECONDS (60) of wall-clock Python stacks
#            from every stage process once steady (py-spy, 25 Hz)
#            -> logs/pp8/profiles/pyspy-bridge<NAME>-<pid>.txt, and the stage of each pid in
#            logs/pp8/profiles/pyspy-bridge<NAME>-ranks.txt
# timers and pyspy are diagnosis: their step times are not results.
#
# Both recompute every layer and accumulate gradients in bf16 (fp32 master weights and Adam), as
# in the TP4xPP2 runs. Under a pipeline Megatron-Bridge pads every row to 2048 on its own.
# STEPS (default 50).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RDSP=$(cd "$HERE/../.." && pwd)
export MODEL=Qwen3.5-27B RECIPE=qwen35_vl_27b_sft_16gpu_h100_bf16_config
export LOGS=$HERE/logs/pp8; mkdir -p "$LOGS"
RUNS=$RDSP/bench/common/runs.sh
DATA=${DATA:-exported:$HOME/mimo/cord_steps-$MODEL}   # DATA: other rows (e.g. MedPix)
# MEG_RC: Bridge's recompute overrides (default: every layer); RECOMPUTE=0: none
MEG_RC=${MEG_RC:-"model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1"}
[ "${RECOMPUTE:-1}" = 0 ] && MEG_RC=""
MEG_BF16=$(cat "$RDSP/bench/common/meg_extra_bf16.txt")

case "${1:-}" in
  bridge)
    GRAD_REDUCE_FP32=false MEGATRON_EXTRA="$MEG_RC $MEG_BF16" bash "$RUNS" tp1pp8-megatron || rc=$?
    mv "$LOGS/tp1pp8-megatron.log" "$LOGS/bridge${NAME:-}.log"  # kept when the run fails
    exit "${rc:-0}" ;;
  timers)
    STEPS=${STEPS:-15} GRAD_REDUCE_FP32=false \
      MEGATRON_EXTRA="$MEG_RC $MEG_BF16 logger.timing_log_level=2 logger.timing_log_option=all" \
      bash "$RUNS" tp1pp8-megatron || rc=$?
    mv "$LOGS/tp1pp8-megatron.log" "$LOGS/timers-bridge${NAME:-}.log"
    exit "${rc:-0}" ;;
  pyspy)
    PROF=$LOGS/profiles; mkdir -p "$PROF"; rm -f "$LOGS/tp1pp8-megatron.log"
    STEPS=${STEPS:-15} GRAD_REDUCE_FP32=false MEGATRON_EXTRA="$MEG_RC $MEG_BF16" \
      bash "$RUNS" tp1pp8-megatron > /dev/null 2>&1 &
    RUN=$!
    until grep -q "iteration        6/" "$LOGS/tp1pp8-megatron.log" 2>/dev/null; do
      kill -0 $RUN 2>/dev/null || { echo "the Bridge run ended before step 6"; break; }
      sleep 5
    done
    # the workers: the launcher and the docker/bash wrappers also name torch.distributed.run
    # (the 8 data workers of each rank are its children and share its command line)
    # `if`, not `&&`: a false last test would fail the assignment and, under set -e, end the script
    workers=$(for p in $(pgrep -f "[r]un_recipe.py"); do
                if tr '\0' ' ' < /proc/$p/cmdline | grep -q torch.distributed.run; then continue; fi
                pp=$(ps -o ppid= -p "$p" | tr -d ' ')
                if tr '\0' ' ' < "/proc/$pp/cmdline" 2>/dev/null | grep -q torch.distributed.run; then echo "$p"; fi
              done)
    for p in $workers; do  # RANK = pipeline stage (TP1, DP1)
      echo "$p $(sudo cat /proc/$p/environ | tr '\0' '\n' | grep -E '^(RANK|LOCAL_RANK)=' | tr '\n' ' ')"
    done | tee "$PROF/pyspy-bridge${NAME:-}-ranks.txt"
    jobs=()
    for p in $workers; do
      sudo "$HOME/venv/bin/py-spy" record --pid "$p" --rate 25 --duration "${PYSPY_SECONDS:-60}" --idle \
        --format raw -o "$PROF/pyspy-bridge${NAME:-}-$p.txt" > "$PROF/pyspy-bridge${NAME:-}-$p.log" 2>&1 &
      jobs+=($!)
    done
    [ ${#jobs[@]} -gt 0 ] && wait "${jobs[@]}"   # only the samplers: the training run keeps going
    wait $RUN || rc=$?
    mv "$LOGS/tp1pp8-megatron.log" "$LOGS/pyspy-bridge${NAME:-}.log"
    exit "${rc:-0}" ;;
  rdsp)
    (unset LD_LIBRARY_PATH && . "$HOME/venv/bin/activate" && . "$HOME/cuda_env.sh" && \
     env DS_BUILD_OPS=0 HF_HOME=$HOME/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
         DATA=$DATA bash "$HERE/rdsp_pp8.sh" ${RDSP_ARGS:-}) 2>&1 | tee "$LOGS/rdsp${NAME:-}.log" ;;
  *) sed -n 4,20p "$0"; exit 1 ;;
esac
