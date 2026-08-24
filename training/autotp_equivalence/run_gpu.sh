#!/bin/bash
# Check that AutoTP does not change what a model computes, on accelerators.
#
# Runs the same steps at three tensor-parallel widths and diffs each sharded run
# against the unsharded one:
#
#   AutoTP=1  the baseline
#   AutoTP=3  an *uneven* split of Qwen3's 16 attention / 8 KV heads (6/6/4 per
#             rank) -- the case a shard-boundary bug is most likely to hit
#   AutoTP=4  an *even* split (4/4/4/4) -- the control. It divides every
#             dimension exactly, so its gap to the baseline is pure
#             floating-point reassociation. If AutoTP=3 drifts no more than
#             AutoTP=4 does, the uneven split is not adding error of its own.
#
# Usage: bash run_gpu.sh [steps] [four comma-separated gpu ids]
set -e
# Overridable so a CPU and a GPU run can proceed at the same time.
MASTER_PORT=${MASTER_PORT:-29500}
STEPS=${1:-500}
GPUS=${2:-0,1,2,3}

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if [ "${#GPU_LIST[@]}" -ne 4 ]; then
    echo "need exactly 4 gpu ids, got '${GPUS}'" >&2
    exit 1
fi
GPUS_1="${GPU_LIST[0]}"
GPUS_3="${GPU_LIST[0]},${GPU_LIST[1]},${GPU_LIST[2]}"
GPUS_4="${GPUS}"

rm -rf runs/gpu

for tp in "1:${GPUS_1}" "3:${GPUS_3}" "4:${GPUS_4}"; do
    deepspeed --master_port "${MASTER_PORT}" --include "localhost:${tp#*:}" train.py \
        --deepspeed_config "configs/autotp${tp%%:*}.json" \
        --metrics_file "runs/gpu/autotp${tp%%:*}.jsonl" --steps "${STEPS}"
done

echo
echo "===== AutoTP=3 (uneven 6/6/4) vs AutoTP=1 ====="
python compare_loss.py runs/gpu/autotp1.jsonl runs/gpu/autotp3.jsonl --print-every 100

echo
echo "===== AutoTP=4 (even 4/4/4/4, control) vs AutoTP=1 ====="
python compare_loss.py runs/gpu/autotp1.jsonl runs/gpu/autotp4.jsonl --print-every 100
