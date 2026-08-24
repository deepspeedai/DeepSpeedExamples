#!/bin/bash
# Check that AutoTP does not change what a model computes, on CPU.
#
# Same comparison as run_gpu.sh but with the gloo backend and no accelerator, so
# it also covers the CPU collectives path and runs anywhere. Slower per step, so
# a shorter --seq_len keeps it practical.
#
#   AutoTP=1  the baseline
#   AutoTP=3  an *uneven* split of Qwen3's 16 attention / 8 KV heads (6/6/4 per
#             rank) -- the case a shard-boundary bug is most likely to hit
#   AutoTP=4  an *even* split (4/4/4/4) -- the control. It divides every
#             dimension exactly, so its gap to the baseline is pure
#             floating-point reassociation. If AutoTP=3 drifts no more than
#             AutoTP=4 does, the uneven split is not adding error of its own.
#
# Usage: bash run_cpu.sh [steps]
set -e
# Overridable so a CPU and a GPU run can proceed at the same time.
MASTER_PORT=${MASTER_PORT:-29500}
STEPS=${1:-500}
export DS_ACCELERATOR=cpu
# Without a cap each rank opens a thread pool sized for the whole machine, and
# several of them oversubscribe it badly -- 27s per step here versus 1.5s. Every
# run uses the same cap so the only difference between them stays the sharding.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}

rm -rf runs/cpu

for tp in 1 3 4; do
    deepspeed --master_port "${MASTER_PORT}" --num_accelerators "${tp}" train.py \
        --deepspeed_config "configs/autotp${tp}.json" \
        --metrics_file "runs/cpu/autotp${tp}.jsonl" --steps "${STEPS}" --seq_len 64
done

echo
echo "===== AutoTP=3 (uneven 6/6/4) vs AutoTP=1 ====="
python compare_loss.py runs/cpu/autotp1.jsonl runs/cpu/autotp3.jsonl --print-every 100

echo
echo "===== AutoTP=4 (even 4/4/4/4, control) vs AutoTP=1 ====="
python compare_loss.py runs/cpu/autotp1.jsonl runs/cpu/autotp4.jsonl --print-every 100
