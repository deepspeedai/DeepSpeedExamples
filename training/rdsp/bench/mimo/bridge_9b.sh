#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# bridge_9b.sh <step>: one step of the 9B Megatron-Bridge vs rdsp TP4xPP2xDP1 comparison, on the
# H100 box, after bench/setup_node.sh; its prep step is also the MIMO 9B runs' (mimo_9b.sh).
# Logs go to logs/9b_tp4pp2dp1/ (and ~/mimo for the container).
#
#   prep        HF -> Megatron checkpoint, then export Bridge's samples (pad 2048)
#   preflight   3 steps of each system: fits in memory? (peak memory in the logs)
#   bridge      Megatron-Bridge, 50 steps
#   rdsp        rdsp, equal work (every row padded to 2048), 50 steps
#   timers      15 steps each, per-phase timers (Megatron timing_log_level=2, rdsp
#               --profile) and per-second GPU utilization (nvidia-smi dmon)
#   nsys        15 steps each under Nsight Systems: NSYS_DELAY_BRIDGE / NSYS_DELAY_RDSP
#               seconds from launch, NSYS_DURATION seconds; reports in logs/9b_tp4pp2dp1/profiles
#   backtrace   like nsys, plus the Python call stack of every CPU-GPU sync
#               (reports logs/9b_tp4pp2dp1/profiles/{bridge,rdsp}-bt.nsys-rep)
#   pyspy       (or pyspy-bridge / pyspy-rdsp) 15 steps each, untraced; 40 s of
#               wall-clock Python stacks from every GPU process once steady (py-spy,
#               25 Hz): logs/9b_tp4pp2dp1/profiles/pyspy-<system>-<pid>.txt
# Profiled runs are diagnosis: their step times are not the results.
#
# RECOMPUTE (default 0) recomputes every layer on BOTH systems (Megatron: full,
# uniform, 1 layer per chunk; rdsp: recompute=1 on both stages). COMPILE_VISION
# (default 0) is rdsp's torch.compile of the vision encoder.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RDSP=$(cd "$HERE/../.." && pwd)
export MODEL=Qwen3.5-9B RECIPE=qwen35_vl_9b_sft_4gpu_h100_bf16_config
export LOGS=$HERE/logs/9b_tp4pp2dp1; mkdir -p "$LOGS"
RUNS=$RDSP/bench/common/runs.sh
DATA=exported:$HOME/mimo/cord_steps-$MODEL
# 9B fits without recompute. Its vision encoder is 27B's (MLP 4304 wide): compiled
# under TP4 it crashed on 27B ("misaligned address"), so it runs eager here too.
export RECOMPUTE=${RECOMPUTE:-0} COMPILE_VISION=${COMPILE_VISION:-0}
MEG_RC=""
[ "${RECOMPUTE:-0}" = 1 ] && MEG_RC="model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1"
TAG=""; [ "${RECOMPUTE:-0}" = 1 ] && TAG=-rc

container() {  # the same NeMo container runs.sh uses
  sudo docker run --gpus all --rm --ipc=host --shm-size=64g --ulimit memlock=-1 \
    -v "$HOME/hf:/root/.cache/huggingface" -v "$HOME/mimo:/workspace" -v "$RDSP:/rdsp" \
    -e HF_HOME=/root/.cache/huggingface -e PYTHONUNBUFFERED=1 -w /opt/Megatron-Bridge \
    nvcr.io/nvidia/nemo:26.08 bash -c "$*"
  sudo chown -R "$(id -u):$(id -g)" "$HOME/hf" "$HOME/mimo"
}
PROF=$LOGS/profiles; mkdir -p "$PROF"
nsys_cmd() {  # nsys_cmd <report name>: a profiler prefix when NSYS_DELAY is set
  [ -n "${NSYS_DELAY:-}" ] || return 0
  echo "nsys profile -t cuda,nvtx ${NSYS_FLAGS:--s none --cpuctxsw=none} --delay $NSYS_DELAY --duration ${NSYS_DURATION:-60} --force-overwrite true -o $PROF/$1"
}
dmon() { nvidia-smi dmon -s u -d 1 -o T > "$LOGS/$1.dmon" 2>&1 & DMON=$!; }
sample() {  # sample <system> <pids...>: 40 s of wall-clock Python stacks per process, at once
  local sys=$1; shift
  local jobs=()
  for p in "$@"; do
    sudo "$HOME/venv/bin/py-spy" record --pid "$p" --rate 25 --duration 40 --idle --format raw \
      -o "$PROF/pyspy-$sys-$p.txt" > "$PROF/pyspy-$sys-$p.log" 2>&1 &
    jobs+=($!)
  done
  wait "${jobs[@]}"   # only the samplers: the training run keeps going
}
rdsp() {  # rdsp <log name> [env...]
  local name=$1; shift
  (unset LD_LIBRARY_PATH && . "$HOME/venv/bin/activate" && . "$HOME/cuda_env.sh" && \
   env DS_BUILD_OPS=0 HF_HOME=$HOME/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       DATA=$DATA "$@" $(nsys_cmd "$name") bash "$HERE/rdsp_9b_tp4pp2.sh" ${RDSP_ARGS:-}) 2>&1 | tee "$LOGS/$name.log"
}

case "${1:-}" in
  prep)  # Bridge's checkpoint only: this layout has no MegatronMIMO run
    container "python scripts/conversion/setup_conversion.py import --hf-model Qwen/$MODEL \
      --megatron-path /workspace/std/$MODEL" 2>&1 | tee "$LOGS/convert.log"
    bash "$RUNS" export ;;
  preflight)
    STEPS=3 MEGATRON_EXTRA="$MEG_RC" bash "$RUNS" tp4pp2-megatron || true
    mv "$LOGS/tp4pp2-megatron.log" "$LOGS/preflight-bridge$TAG.log"
    # --profile: per-phase timers and peak memory per GPU (a 3-step memory check, not timing)
    RDSP_ARGS=--profile rdsp "preflight-rdsp$TAG" STEPS=3 RECOMPUTE="${RECOMPUTE:-0}" || true ;;
  preflight-rdsp)  # rdsp only; NAME suffixes the log, e.g. -nocompile
    RDSP_ARGS=--profile rdsp "preflight-rdsp$TAG${NAME:-}" STEPS=3 RECOMPUTE="${RECOMPUTE:-0}" \
      COMPILE_VISION="${COMPILE_VISION:-1}" || true ;;
  bridge)
    MEGATRON_EXTRA="$MEG_RC" bash "$RUNS" tp4pp2-megatron
    mv "$LOGS/tp4pp2-megatron.log" "$LOGS/bridge$TAG.log" ;;
  rdsp)     rdsp "rdsp-equal$TAG" PAD=equal RECOMPUTE="${RECOMPUTE:-0}" ;;
  timers)
    dmon timers-bridge; STEPS=15 MEGATRON_EXTRA="$MEG_RC logger.timing_log_level=2" bash "$RUNS" tp4pp2-megatron || true
    kill $DMON; mv "$LOGS/tp4pp2-megatron.log" "$LOGS/timers-bridge.log"
    dmon timers-rdsp; RDSP_ARGS=--profile rdsp timers-rdsp STEPS=15 || true; kill $DMON ;;
  nsys)
    STEPS=15 NSYS_DELAY=$NSYS_DELAY_BRIDGE NSYS_DURATION=${NSYS_DURATION:-60} MEGATRON_EXTRA="$MEG_RC" \
      bash "$RUNS" tp4pp2-megatron || true
    mv "$LOGS/tp4pp2-megatron.log" "$LOGS/nsys-bridge.log"
    mv "$HOME/mimo/nsys/tp4pp2-megatron.nsys-rep" "$PROF/bridge.nsys-rep" || true
    NSYS_DELAY=$NSYS_DELAY_RDSP rdsp nsys-rdsp STEPS=15 || true
    mv "$PROF/nsys-rdsp.nsys-rep" "$PROF/rdsp.nsys-rep" || true ;;
  backtrace)
    export NSYS_FLAGS="--sample=process-tree --cpuctxsw=none --cudabacktrace=sync --python-backtrace=cuda"
    STEPS=15 NSYS_DELAY=$NSYS_DELAY_BRIDGE NSYS_DURATION=${NSYS_DURATION:-45} MEGATRON_EXTRA="$MEG_RC" \
      bash "$RUNS" tp4pp2-megatron || true
    mv "$LOGS/tp4pp2-megatron.log" "$LOGS/bt-bridge.log"
    mv "$HOME/mimo/nsys/tp4pp2-megatron.nsys-rep" "$PROF/bridge-bt.nsys-rep" || true
    NSYS_DELAY=$NSYS_DELAY_RDSP rdsp bt-rdsp STEPS=15 || true
    mv "$PROF/bt-rdsp.nsys-rep" "$PROF/rdsp-bt.nsys-rep" || true ;;
  pyspy) bash "$0" pyspy-bridge && bash "$0" pyspy-rdsp ;;
  pyspy-bridge)
    STEPS=15 MEGATRON_EXTRA="$MEG_RC" bash "$RUNS" tp4pp2-megatron > /dev/null 2>&1 &
    RUN=$!
    until grep -q "iteration        6/" "$LOGS/tp4pp2-megatron.log" 2>/dev/null; do sleep 5; done
    workers=$(for p in $(pgrep -f "[r]un_recipe.py"); do
                tr '\0' ' ' < /proc/$p/cmdline | grep -q torch.distributed.run || echo "$p"; done)
    echo "bridge processes:" $workers
    sample bridge $workers
    wait $RUN; mv "$LOGS/tp4pp2-megatron.log" "$LOGS/pyspy-bridge.log" ;;
  pyspy-rdsp)
    rdsp pyspy-rdsp STEPS=15 > /dev/null &
    RUN=$!
    until grep -q "^step 5 loss" "$LOGS/pyspy-rdsp.log" 2>/dev/null; do sleep 5; done
    actors=$(pgrep -f "[r]ay::StageWorkerActor")
    echo "rdsp actors:" $actors
    sample rdsp $actors
    wait $RUN ;;
  *) sed -n 4,21p "$0"; exit 1 ;;
esac
