#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# check_bitexact.sh -- verify that Reflow is bit-identical to ZeRO-Infinity
# (optimizer offload + ZeRO Stage 3).
#
# Usage:  bash check_bitexact.sh [num_gpus]        (default: 1)
#
# Runs the same OPT-350M job twice, once with each mode, under full
# determinism (--loss_check), then diffs the per-step hex-float losses.
# Everything is self-contained: the two configs are written here and
# differ only in the Reflow keys.
#
# --loss_check pins the NCCL collectives and forces deterministic
# algorithms, so this is slower than a normal run. It is for
# verification only -- speed and memory are measured by run_compare.sh.

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

NUM_GPUS=${1:-1}
BATCH_SIZE=${BATCH_SIZE:-4}
OPTIMIZER=${OPTIMIZER:-adam}
BENCH_STEPS=${BENCH_STEPS:-6}
MODEL_NAME=${MODEL_NAME:-facebook/opt-350m}

# CPU affinity knobs, as in the finetune_*.sh scripts. Keep
# main_thread_cores + bucketwise_cores_per_worker within the cores
# visible to each rank.
MAIN_CORES=${MAIN_CORES:-2}
WORKER_CORES=${WORKER_CORES:-8}

OUT_DIR=${OUT_DIR:-bitexact_out}
for value in "$NUM_GPUS" "$BATCH_SIZE" "$BENCH_STEPS" "$MAIN_CORES" "$WORKER_CORES"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "GPU, batch, step and core counts must be positive integers" >&2
        exit 2
    fi
done
mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------- configs
# Same template as the finetune_*.sh scripts. overlap_comm stays false:
# with overlapped reductions the reduction order becomes timing-dependent
# and the two modes can legitimately diverge in the last bits.
write_config() {   # $1 = path, $2 = reflow|zerooffload
    if [ "$2" = "reflow" ]; then
        REFLOW_BLOCK=',
        "reflow": {
            "enable_cpu_affinity": true,
            "bucketwise_cores_per_worker": '"$WORKER_CORES"',
            "main_thread_cores": '"$MAIN_CORES"'
        }'
    else
        REFLOW_BLOCK=""
    fi
    cat > "$1" << EOF
{
    "train_micro_batch_size_per_gpu": $BATCH_SIZE,
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 0.0,
    "bf16": { "enabled": true },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": false,
        "reduce_bucket_size": 4e8,
        "sub_group_size": 4e8,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }$REFLOW_BLOCK
    },
    "wall_clock_breakdown": true
}
EOF
}

REFLOW_CFG="$OUT_DIR/opt-350m_reflow_config.json"
ZERO_CFG="$OUT_DIR/opt-350m_zerooffload_config.json"
write_config "$REFLOW_CFG" reflow
write_config "$ZERO_CFG"   zerooffload

# ------------------------------------------------------------------- runs
# Eager attention is required: flash-attention's backward is
# non-deterministic and would mask a real difference.
COMMON=(--model_name "$MODEL_NAME"
    --optimizer "$OPTIMIZER" --loss_check --attn_implementation eager
    --lr 1e-5 --batch_size "$BATCH_SIZE" --max_length 512
    --bench_steps "$BENCH_STEPS" --warmup_steps 0 --output_dir "$OUT_DIR/out")

echo "[1/2] Reflow ..."
deepspeed --num_gpus="$NUM_GPUS" --bind_cores_to_rank "$SCRIPT_DIR/finetune_zero3.py" \
    --deepspeed_config="$REFLOW_CFG" \
    "${COMMON[@]}" > "$OUT_DIR/reflow.log" 2>&1

echo "[2/2] ZeRO-Infinity baseline ..."
deepspeed --num_gpus="$NUM_GPUS" --bind_cores_to_rank "$SCRIPT_DIR/finetune_zero3.py" \
    --deepspeed_config="$ZERO_CFG" \
    "${COMMON[@]}" > "$OUT_DIR/zero.log" 2>&1

# Ignore timestamps but require both jobs to have completed every requested step.
python3 - "$OUT_DIR/reflow.log" "$OUT_DIR/zero.log" "$BENCH_STEPS" <<'PYTHON'
import re
import sys
from pathlib import Path

expected_steps = list(range(1, int(sys.argv[3]) + 1))
losses = []
for filename in sys.argv[1:3]:
    matches = re.findall(r"BITLOSS step (\d+) hex=(\S+)", Path(filename).read_text())
    if [int(step) for step, _ in matches] != expected_steps:
        sys.exit(f"Incomplete loss log: {filename}; expected steps 1 through {expected_steps[-1]}")
    losses.append([value for _, value in matches])
if losses[0] != losses[1]:
    sys.exit(f"MISMATCH -- losses differ; see {sys.argv[1]} and {sys.argv[2]}")
print("BIT-IDENTICAL")
PYTHON
