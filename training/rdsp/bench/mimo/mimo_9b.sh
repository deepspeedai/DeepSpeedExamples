#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# mimo_9b.sh <step> [layout] [steps]: MegatronMIMO vs rdsp on Qwen3.5-9B, 8x H100, on the box
# after bench/setup_node.sh and the 9B export (bridge_9b.sh prep). Logs in logs/<layout>/.
#
#   convert              HF -> MegatronMIMO checkpoint for 9B (once)
#   mimo <layout> [n]    MegatronMIMO, n steps (50)      -> logs/<layout>/mimo.log
#   rdsp <layout> [n]    rdsp, the same grids, n steps   -> logs/<layout>/rdsp.log
#   sweep <layout>       MegatronMIMO, 15 steps          -> logs/sweep/<layout>.log
#   best <language>      the fastest vision layout for that language grid (median of
#                        steps 6-15, MIMO), counting the 1-GPU vision run from its 50-step log
#   pyspy <layout>       20 steps each, untraced; 40 s of wall-clock Python stacks from every
#                        GPU process once steady (py-spy, 25 Hz) -> logs/<layout>/profiles/pyspy-*.txt;
#                        also records each system's launch-to-steady time for nsys
#   nsys <layout>        20 steps each under Nsight Systems, a window from that steady time
#                        -> logs/<layout>/profiles/{mimo,rdsp}.nsys-rep and .sqlite
#   backtrace <layout>   rdsp only, 12 steps under Nsight Systems with the call stack of every
#                        kernel launch (names the autograd node or Python line behind each GPU
#                        operation); needs the pyspy step's steady time
#                        -> logs/<layout>/profiles/rdsp-bt.nsys-rep and .sqlite
# Profiled runs are diagnosis: their step times are not the results. TAG (e.g. -nov11) suffixes
# the profiles directory and the profiled runs' logs, so a later run keeps the earlier files.
#
# Layout name: 9b_<vision>_<language>. Language on ranks 0-3, the vision encoder after it.
#   vision:   v1 (1 GPU), vtp2, vtp4, vdp2, vtp2dp2
#   language: tp2pp2dp1 (1 row per microbatch, 64 microbatches)
#             tp2pp1dp2 (2 rows per microbatch, 1 per DP copy, 32 microbatches)
# Both systems: every row padded to 2048 (MIMO's default; rdsp --pad-multiple 0), MIMO's
# per-microbatch loss mean (rdsp --loss microbatch-mean), no recompute, rdsp's vision
# encoder eager (no torch.compile), 64 rows per step.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RDSP=$(cd "$HERE/../.." && pwd)
L=$HERE/logs
export MODEL=Qwen3.5-9B
RUNS=$RDSP/bench/common/runs.sh
DATA=exported:$HOME/mimo/cord_steps-$MODEL

spec() {  # spec <layout>: sets IMG, VGPUS, VSTAGE, LANG, MBS, ROWS, MICRO, STAGES
  local v=${1#9b_}; local l=${v#*_}; v=${v%%_*}
  case $v in
    v1)      IMG=tp=1,pp=1,dp=1; VGPUS=1; VSTAGE=0:gpus=1,zero=1 ;;
    vtp2)    IMG=tp=2,pp=1,dp=1; VGPUS=2; VSTAGE=0:gpus=2,tp=2,zero=1 ;;
    vtp4)    IMG=tp=4,pp=1,dp=1; VGPUS=4; VSTAGE=0:gpus=4,tp=4,zero=1 ;;
    vdp2)    IMG=tp=1,pp=1,dp=2; VGPUS=2; VSTAGE=0:gpus=2,zero=1 ;;
    vtp2dp2) IMG=tp=2,pp=1,dp=2; VGPUS=4; VSTAGE=0:gpus=4,tp=2,zero=1 ;;
    *) echo "unknown vision layout $v" >&2; exit 1 ;;
  esac
  case $l in
    tp2pp2dp1) LANG=tp=2,pp=2,dp=1; MBS=1; ROWS=1; MICRO=64
               STAGES="--stages 3 --cuts 0,balanced --stage $VSTAGE --stage 1:gpus=2,tp=2,zero=1 --stage 2:gpus=2,tp=2,zero=1" ;;
    tp2pp1dp2) LANG=tp=2,pp=1,dp=2; MBS=2; ROWS=2; MICRO=32
               STAGES="--stages 2 --cuts 0 --stage $VSTAGE --stage 1:gpus=4,tp=2,zero=1" ;;
    *) echo "unknown language layout $l" >&2; exit 1 ;;
  esac
}

mimo_run() {  # mimo_run <layout> <steps> <log dir> <log name>; NSYS_DELAY/NSYS_DURATION trace it
  spec "$1"
  mkdir -p "$3"
  env LOGS="$3" STEPS="$2" MIMO_NAME="$4" MIMO_GPUS=$((4 + VGPUS)) MIMO_MBS=$MBS \
      MIMO_LANGUAGE="$LANG,rank_offset=0" MIMO_IMAGES="$IMG,rank_offset=4" bash "$RUNS" mimo-layout
}

rdsp_run() {  # rdsp_run <layout> <steps> <log name> [profiler prefix...]
  local layout=$1 steps=$2 name=$3; shift 3
  spec "$layout"; mkdir -p "$L/$layout"
  (unset LD_LIBRARY_PATH && . "$HOME/venv/bin/activate" && . "$HOME/cuda_env.sh" && \
   env DS_BUILD_OPS=0 HF_HOME=$HOME/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   "$@" python "$RDSP/train_vl.py" --model "Qwen/$MODEL" --dataset "$DATA" $STAGES \
     --rows $ROWS --microbatches $MICRO --seq 2048 --pad-multiple 0 \
     --steps "$steps" --lr 1e-5 --betas 0.9,0.95 --eps 1e-8 --weight-decay 0.0 \
     --grad-dtype fp32 --loss microbatch-mean --liger \
     --prefetch --sharded-loss --drop-padding-mask) 2>&1 | tee "$L/$layout/$name.log"
}

sample() {  # sample <out dir> <system> <pids...>: 40 s of wall-clock Python stacks per process, at once
  local out=$1 sys=$2; shift 2
  local jobs=()
  for p in "$@"; do
    sudo "$HOME/venv/bin/py-spy" record --pid "$p" --rate 25 --duration 40 --idle --format raw \
      -o "$out/pyspy-$sys-$p.txt" > "$out/pyspy-$sys-$p.log" 2>&1 &
    jobs+=($!)
  done
  wait "${jobs[@]}"   # only the samplers: the training run keeps going
}

steady() {  # steady <log> <regex> <start epoch> <run pid>: wait for the line, print seconds since start
  until grep -qE "$2" "$1" 2>/dev/null; do kill -0 "$4" 2>/dev/null || return 1; sleep 2; done
  echo $(( $(date +%s) - $3 ))
}

case "${1:-}" in
  convert)
    mkdir -p "$L"
    sudo docker run --gpus all --rm --ipc=host --shm-size=64g --ulimit memlock=-1 \
      -v "$HOME/hf:/root/.cache/huggingface" -v "$HOME/mimo:/workspace" \
      -e HF_HOME=/root/.cache/huggingface -w /opt/Megatron-Bridge nvcr.io/nvidia/nemo:26.08 bash -c \
      "python -m torch.distributed.run --nproc_per_node=2 examples/conversion/convert_megatron_mimo.py \
         import --hf-model Qwen/$MODEL --megatron-path /workspace/$MODEL-mimo \
         --component language=tp=1,dp=1,rank_offset=0 --component images=tp=1,dp=1,rank_offset=1 \
         --torch-dtype bfloat16" 2>&1 | tee "$L/convert.log"
    sudo chown -R "$(id -u):$(id -g)" "$HOME/hf" "$HOME/mimo" ;;
  mimo)  mimo_run "$2" "${3:-50}" "$L/$2" mimo ;;
  sweep) mimo_run "$2" 15 "$L/sweep" "$2" || true ;;
  rdsp)  rdsp_run "$2" "${3:-50}" rdsp ;;
  pyspy)
    T=${TAG:-}; P=$L/$2/profiles$T; D=$L/$2; mkdir -p "$P" "$D"; rm -f "$D/pyspy-mimo$T.log" "$D/pyspy-rdsp$T.log"
    t0=$(date +%s); mimo_run "$2" 20 "$D" "pyspy-mimo$T" > /dev/null 2>&1 &
    RUN=$!
    steady "$D/pyspy-mimo$T.log" "iteration +6/" "$t0" $RUN > "$P/steady-mimo.s"
    # the workers: the launcher and the docker/bash wrappers also name torch.distributed.run
    workers=$(for p in $(pgrep -f "[f]inetune_qwen35_vl.py"); do
                tr '\0' ' ' < /proc/$p/cmdline | grep -q torch.distributed.run || echo "$p"; done)
    echo "mimo processes:" $workers "(steady after $(cat "$P/steady-mimo.s") s)"
    sample "$P" mimo $workers
    wait $RUN || true
    t0=$(date +%s); rdsp_run "$2" 20 "pyspy-rdsp$T" > /dev/null 2>&1 &
    RUN=$!
    steady "$D/pyspy-rdsp$T.log" "^step 5 loss" "$t0" $RUN > "$P/steady-rdsp.s"
    actors=$(pgrep -f "[r]ay::StageWorkerActor")
    echo "rdsp actors:" $actors "(steady after $(cat "$P/steady-rdsp.s") s)"
    sample "$P" rdsp $actors
    wait $RUN || true ;;
  nsys)  # a window from each system's steady time (pyspy step), NSYS_DURATION seconds (75)
    T=${TAG:-}; P=$L/$2/profiles$T; D=$L/$2; mkdir -p "$HOME/mimo/nsys"
    NSYS_DELAY=$(cat "$P/steady-mimo.s") NSYS_DURATION=${NSYS_DURATION:-75} \
      mimo_run "$2" 20 "$D" "nsys-mimo$T" || true
    mv "$HOME/mimo/nsys/nsys-mimo$T.nsys-rep" "$P/mimo.nsys-rep" || true
    rdsp_run "$2" 20 "nsys-rdsp$T" nsys profile -t cuda,nvtx -s none --cpuctxsw=none \
      --delay "$(cat "$P/steady-rdsp.s")" --duration "${NSYS_DURATION:-75}" --force-overwrite true \
      -o "$P/rdsp" || true
    # the container's nsys is newer than the host's: export both reports there
    sudo docker run --gpus all --ipc=host -v "$P:/p" nvcr.io/nvidia/nemo:26.08 bash -c \
      "for r in mimo rdsp; do nsys export --type sqlite --force-overwrite true -o /p/\$r.sqlite /p/\$r.nsys-rep; done"
    sudo chown -R "$(id -u):$(id -g)" "$P" ;;
  backtrace)
    T=${TAG:-}; P=$L/$2/profiles$T; mkdir -p "$P"
    # --kill none: the run ends by itself after the window, so no process loses its last records
    rdsp_run "$2" 12 "bt-rdsp$T" nsys profile -t cuda,nvtx --sample=process-tree --cpuctxsw=none \
      --cudabacktrace=kernel --kill none --delay "$(cat "$P/steady-rdsp.s")" \
      --duration "${NSYS_DURATION:-40}" --force-overwrite true -o "$P/rdsp-bt" || true
    # the host's nsys made the report; the container's may be older (2026-11-11) or newer (October)
    nsys export --type sqlite --force-overwrite true -o "$P/rdsp-bt.sqlite" "$P/rdsp-bt.nsys-rep" || {
      sudo docker run --rm --gpus all --ipc=host -v "$P:/p" nvcr.io/nvidia/nemo:26.08 bash -c \
        "nsys export --type sqlite --force-overwrite true -o /p/rdsp-bt.sqlite /p/rdsp-bt.nsys-rep"
      sudo chown -R "$(id -u):$(id -g)" "$P"; } ;;
  best)
    python3 - "$L" "$2" <<'PY'
import glob, os, re, statistics, sys
logs_dir, lang = sys.argv[1], sys.argv[2]
logs = {f"9b_v1_{lang}": f"{logs_dir}/9b_v1_{lang}/mimo.log"}
logs.update({os.path.basename(p)[:-4]: p for p in glob.glob(f"{logs_dir}/sweep/9b_*_{lang}.log")})
med = {}
for name, path in logs.items():
    t = [float(m[2]) for m in re.finditer(r"iteration +(\d+)/ +\d+ .*?per iteration \(ms\): ([\d.]+)",
                                         open(path).read()) if 6 <= int(m[1]) <= 15]
    if len(t) >= 5:
        med[name] = statistics.median(t) / 1e3
for name, s in sorted(med.items(), key=lambda kv: kv[1]):
    print(f"{name} {s:.2f}", file=sys.stderr)
print(min(med, key=med.get))
PY
    ;;
  *) sed -n 2,26p "$0"; exit 1 ;;
esac
