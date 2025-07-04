#!/bin/bash

NGPUS=2
HIDDEN_SIZE=4096
NUM_LAYERS=4
TRIALS=1

PIN_MEMORY_OPTS=(0 1)
TOPK_RATIOS=(0.1 0.2)
UPDATE_INTERVALS=(2 4)
OVERLAP_STEPS=(1 0)

for pin_memory in "${PIN_MEMORY_OPTS[@]}"; do
  for topk in "${TOPK_RATIOS[@]}"; do
    for update in "${UPDATE_INTERVALS[@]}"; do
      for overlap in "${OVERLAP_STEPS[@]}"; do
        for ((trial=0; trial<$TRIALS; trial++)); do
          # Generate a random port between 20000 and 65000
          MASTER_PORT=$((20000 + RANDOM % 45000))
          echo "[Trial $((trial+1))] pin_memory=$pin_memory, topk=$topk, update=$update, overlap_step=$overlap (MASTER_PORT=$MASTER_PORT)" | tee -a zf_benchmark.log
          deepspeed --master_port $MASTER_PORT \
            --num_gpus=$NGPUS \
            zf_benchmark.py \
            --hidden_dim $HIDDEN_SIZE \
            --nlayers $NUM_LAYERS \
            --iteration 5 \
            --pin_memory_opts $pin_memory \
            --topk_ratios $topk \
            --update_intervals $update \
            --overlap_steps $overlap | tee -a zf_benchmark.log
        done
      done
    done
  done
done
python output_table.py
