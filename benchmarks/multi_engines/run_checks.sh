#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
#
# Run every multi-engine process-group check, one torchrun per case, then summarize.
#
#   bash run_checks.sh                     # all cases, default device and backend, eager init
#   EAGER=0 bash run_checks.sh             # lazy init
#   BACKEND=gloo DEVICE=cpu bash run_checks.sh
#   CASES="registry_path destroy_diff_order" REPEATS=3 bash run_checks.sh
#
# Environment: PYTHON (default python), LOG_DIR (default logs/<timestamp>), CASES, REPEATS (default 1),
# CASE_TIMEOUT seconds per case (default 150), plus DEVICE, BACKEND, EAGER (see common.py).
# Cases that are expected to hang are bounded by CASE_TIMEOUT, so a full run takes 20-30 minutes.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON=${PYTHON:-python}
LOG_DIR=${LOG_DIR:-$HERE/logs/$(date -u +%Y%m%dT%H%M%SZ)}
REPEATS=${REPEATS:-1}
CASE_TIMEOUT=${CASE_TIMEOUT:-150}
CASES=${CASES:-"registry_path destroy_same_order memory_cycles overlap_members_only subset_then_all split_then_all destroy_diff_order shrink_then_all timeout_probe"}
PORT=${MASTER_PORT:-29650}
mkdir -p "$LOG_DIR"
echo "logs: $LOG_DIR"

for i in $(seq 1 "$REPEATS"); do
    for case in $CASES; do
        nproc=4
        [ "$case" = timeout_probe ] && nproc=2
        variants=("$case")
        [ "$case" = shrink_then_all ] && variants=("shrink_then_all:0" "shrink_then_all:1")
        for v in "${variants[@]}"; do
            name=${v/:/_follow}
            follow=0
            [[ "$v" == *:1 ]] && follow=1
            log="$LOG_DIR/${name}_run${i}.log"
            echo "=== ${name} (run ${i})"
            CASE=${v%%:*} FOLLOW=$follow CASE_TIMEOUT=$CASE_TIMEOUT \
                timeout -k 10 $((CASE_TIMEOUT + 60)) "$PYTHON" -m torch.distributed.run \
                --nproc_per_node="$nproc" --master_port=$((PORT++)) "$HERE/multi_engine_checks.py" > "$log" 2>&1
            echo "rc=$?" >> "$log"
        done
    done
done
"$PYTHON" "$HERE/summarize.py" "$LOG_DIR"
