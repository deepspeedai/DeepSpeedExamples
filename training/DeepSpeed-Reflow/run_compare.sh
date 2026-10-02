#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# run_compare.sh -- one-command Reflow vs ZeRO-Infinity comparison.
#
# Usage:
#   bash run_compare.sh                          # Llama-7B and OPT-30B, 1 GPU
#   bash run_compare.sh finetune_opt-30b_1gpu.sh # a specific config
#   bash run_compare.sh script_a.sh script_b.sh  # several configs
#
# For each finetune script it runs both modes back to back, samples GPU
# and host memory while they run, and prints:
#
#   * peak GPU memory and peak host RAM for each mode
#   * steady-state step time and the resulting speedup
#   * TFLOPS, bwd_microstep, step_microstep
#
# Timings are taken from ordinary runs -- never pass --loss_check here,
# since deterministic collectives distort them (bit-exactness is checked
# separately by check_bitexact.sh).
#
# The DeepSpeed config must have "wall_clock_breakdown": true for the
# bwd_microstep / step_microstep timers to appear.
#
# Before running a configuration, free host RAM and free GPU memory are
# checked against what the ZeRO-Infinity baseline needs (the heavier of
# the two modes). A configuration that does not fit is skipped rather
# than left to die part-way through. Set FORCE=1 to run it anyway.

set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
sampler=""
cleanup() {
    if [ -n "$sampler" ]; then
        kill "$sampler" 2>/dev/null || true
        wait "$sampler" 2>/dev/null || true
        sampler=""
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

SCRIPTS=("$@")
if [ ${#SCRIPTS[@]} -eq 0 ]; then
    SCRIPTS=(finetune_llama-7b_1gpu.sh finetune_opt-30b_1gpu.sh)
fi

OUT_DIR=${OUT_DIR:-compare_out}
SAMPLE_SEC=${SAMPLE_SEC:-2}
mkdir -p "$OUT_DIR"

# ------------------------------------------------------------ prechecks
# Requirements of the ZeRO-Infinity baseline (the heavier of the two
# modes), in MiB: our measured peaks plus 1%.
#
#   Llama-7B 1 GPU : measured 132151 host / 31118 GPU
#   OPT-30B  1 GPU : measured 537467 host / 87980 GPU
#
# Configurations we have not measured return "unknown" and are not
# pre-checked; they simply run, and a failure is reported afterwards.
req_host_mib() {
    case "$1" in
        *llama-7b*_1gpu*) echo 133473 ;;   # ~130 GB
        *opt-30b*_1gpu*)  echo 542842 ;;   # ~530 GB
        *opt-350m*)       echo   8192 ;;
        *)                echo unknown ;;
    esac
}
req_gpu_mib() {
    case "$1" in
        *llama-7b*_1gpu*) echo 31429 ;;    # ~31 GB
        *opt-30b*_1gpu*)  echo 88860 ;;    # ~87 GB
        *opt-350m*)       echo  8192 ;;
        *)                echo unknown ;;
    esac
}

free_host_mib() {
    awk '/^MemAvailable:/ {printf "%d", $2/1024}' /proc/meminfo
}
free_gpu_mib() {
    nvidia-smi --query-gpu=memory.total,memory.used \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', *' '{d=$1-$2; if (d>m) m=d} END {printf "%d", m}'
}

# Returns 0 if the configuration fits or is unmeasured, 1 if it clearly
# does not fit (with a reason).
fits() {  # $1 = script name
    local rh rg fh fg
    rh=$(req_host_mib "$1"); rg=$(req_gpu_mib "$1")
    if [ "$rh" != unknown ]; then
        fh=$(free_host_mib)
        if [ "${fh:-0}" -lt "$rh" ]; then
            echo "  SKIPPED: needs ~$((rh/1024)) GB of free host RAM, ${fh:-0} MiB available."
            return 1
        fi
    fi
    if [ "$rg" != unknown ]; then
        fg=$(free_gpu_mib)
        if [ "${fg:-0}" -lt "$rg" ]; then
            echo "  SKIPPED: needs ~$((rg/1024)) GB of free GPU memory, ${fg:-0} MiB available."
            return 1
        fi
    fi
    return 0
}

# -------------------------------------------------------------- sampling
# Peak GPU memory comes from nvidia-smi; peak host RAM is the summed RSS
# of the training processes, which includes the pinned FP32 master and
# optimizer state. (On multi-rank runs shared pages are counted once per
# rank, so this is an upper bound; the 1-GPU configs have a single rank.)
sample_loop() {
    while :; do
        gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
              2>/dev/null | sort -rn | head -1)
        cpu=$(ps -eo rss=,args= | grep -F finetune_zero3.py | grep -v grep \
              | awk '{s+=$1} END {print int(s/1024)}')
        echo "${gpu:-0} ${cpu:-0}"
        sleep "$SAMPLE_SEC"
    done
}

peak_col() {  # $1 = sample file, $2 = column
    awk -v c="$2" '{if ($c+0 > m) m = $c+0} END {printf "%.0f", m}' "$1"
}

# --------------------------------------------------------------- parsing
# Steady state excludes step 1, which absorbs warmup and JIT compilation.
mean_step_time() {
    local summary
    summary=$(sed -n 's/.*Mean step time after [0-9]* warmup steps: \([0-9.]*\)ms/\1/p' "$1" | tail -1)
    if [ -n "$summary" ]; then
        echo "$summary"
        return
    fi
    # Older logs have no summary. Preserve numeric step order for steps 10 and above.
    sed -n 's/.* - finetune_zero3 - INFO - Step *\([0-9]*\) | Loss: [^|]* | Time: *\([0-9]*\)ms.*/\1 \2/p' "$1" \
        | sort -n -k1,1 | tail -n +2 \
        | awk '{s+=$2; n++} END {if (n) printf "%.0f", s/n; else printf "0"}'
}
mean_tflops() {
    grep ' - finetune_zero3 - INFO - Step' "$1" \
        | grep -o 'TFLOPS(w/ recompute): *[0-9.]*' | tail -n +2 \
        | awk -F': *' '{s+=$2; n++} END {if (n) printf "%.1f", s/n; else printf "n/a"}'
}
mean_timer() {  # $2 = timer name
    grep -o "$2: [0-9.]*" "$1" | tail -n +2 \
        | awk -F': ' '{s+=$2; n++} END {if (n) printf "%.1f", s/n; else printf "n/a"}'
}

# ------------------------------------------------------------------ main
failed_any=0
for script in "${SCRIPTS[@]}"; do
    if [[ "$script" != /* && ! -f "$script" ]]; then
        script="$SCRIPT_DIR/$script"
    fi
    if [ ! -f "$script" ]; then
        echo "FAILED: $script not found"
        failed_any=1
        continue
    fi
    tag=$(basename "$script" .sh)
    echo "############################################################"
    echo "# $tag"
    echo "############################################################"

    if [ "${FORCE:-0}" != "1" ] && ! fits "$script"; then
        echo "  Set FORCE=1 to run it anyway."
        echo
        continue
    fi

    failed_logs=()
    for mode in reflow zerooffload; do
        echo "  running $mode ..."
        sample_loop > "$OUT_DIR/$tag.$mode.mem" &
        sampler=$!
        if ! bash "$script" "$mode" > "$OUT_DIR/$tag.$mode.log" 2>&1; then
            failed_logs+=("$OUT_DIR/$tag.$mode.log")
        fi
        cleanup
    done

    rl="$OUT_DIR/$tag.reflow.log";      rm_="$OUT_DIR/$tag.reflow.mem"
    zl="$OUT_DIR/$tag.zerooffload.log"; zm="$OUT_DIR/$tag.zerooffload.mem"

    # A run that died -- out of host RAM or GPU memory, most often -- has
    # no step lines. Say so plainly instead of printing a table of n/a,
    # and move on to the next configuration; the results already printed
    # above remain valid.
    for f in "$rl" "$zl"; do
        grep -q ' - finetune_zero3 - INFO - Step' "$f" || failed_logs+=("$f")
    done
    if [ ${#failed_logs[@]} -gt 0 ]; then
        echo
        failed_any=1
        echo "  FAILED: training exited unsuccessfully or produced no step logs. Last lines:"
        for f in "${failed_logs[@]}"; do
            echo "  --- $f"
            tail -5 "$f" | sed 's/^/    /'
        done
        echo
        continue
    fi

    r_step=$(mean_step_time "$rl"); z_step=$(mean_step_time "$zl")
    speedup=$(awk -v r="$r_step" -v z="$z_step" \
        'BEGIN {if (r > 0) printf "%.2f", z/r; else printf "n/a"}')

    echo
    printf '%-28s %14s %14s\n' "metric" "Reflow" "ZeRO-Inf."
    printf '%-28s %14s %14s\n' "peak GPU memory (MiB)" \
        "$(peak_col "$rm_" 1)" "$(peak_col "$zm" 1)"
    printf '%-28s %14s %14s\n' "peak host RAM (MiB)" \
        "$(peak_col "$rm_" 2)" "$(peak_col "$zm" 2)"
    printf '%-28s %14s %14s\n' "step time (ms)"      "$r_step"  "$z_step"
    printf '%-28s %14s %14s\n' "TFLOPS"              \
        "$(mean_tflops "$rl")" "$(mean_tflops "$zl")"
    printf '%-28s %14s %14s\n' "bwd_microstep (ms)"  \
        "$(mean_timer "$rl" bwd_microstep)"  "$(mean_timer "$zl" bwd_microstep)"
    printf '%-28s %14s %14s\n' "step_microstep (ms)" \
        "$(mean_timer "$rl" step_microstep)" "$(mean_timer "$zl" step_microstep)"
    echo
    echo "  SPEEDUP: ${speedup}x"
    echo "  Logs: $OUT_DIR/$tag.*.log"
    echo
done

exit "$failed_any"
