#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
set -euo pipefail

echo "================================================"
echo "Llama 13B Fine-tuning with DeepSpeed Reflow on 1 GPU"
echo "================================================"

# MODE options: "reflow" or "zerooffload"
MODE=${1:-}
BATCH_SIZE=${2:-4}

GPUS_PER_NODE=1
if [[ "$MODE" != "reflow" && "$MODE" != "zerooffload" ]]; then
    echo "Usage: bash $0 <reflow|zerooffload> [global_batch_size]" >&2
    exit 2
fi
if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || (( BATCH_SIZE % GPUS_PER_NODE != 0 )); then
    echo "global_batch_size must be positive and divisible by $GPUS_PER_NODE" >&2
    exit 2
fi
MICRO_BATCH_SIZE=$((BATCH_SIZE / GPUS_PER_NODE))

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_NAME="meta-llama/Llama-2-13b-hf"
OUTPUT_DIR="${SCRIPT_DIR}/llama-13b_${MODE}_output"
DS_CONFIG_JSON="${SCRIPT_DIR}/llama-13b_${MODE}_config.json"

mkdir -p "$OUTPUT_DIR"

# Script argument parameters
ACTIVATION_CHECKPOINTING=true
SAVE_CHECKPOINT=false
MAX_LENGTH=${MAX_LENGTH:-2048}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
LOG_INTERVAL=1
DATASET_NAME="tatsu-lab/alpaca"
DATASET_PERCENTAGE=10.0
USE_WANDB=false
WANDB_PROJECT="llama-13b"
WANDB_RUN_NAME="llama-13b-$MODE"
DETERMINISTIC=false
BENCH_STEPS=${BENCH_STEPS:-10}
WARMUP_STEPS=${WARMUP_STEPS:-1}

EPOCHS=1
LR=1e-5
WEIGHT_DECAY=0.01
SEED=42

ACTIVATION_CHECKPOINTING_FLAG=""
if [ "$ACTIVATION_CHECKPOINTING" = "true" ]; then
    ACTIVATION_CHECKPOINTING_FLAG="--activation_checkpointing"
fi

SAVE_CHECKPOINT_ARG=""
if [ "$SAVE_CHECKPOINT" = "true" ]; then
    SAVE_CHECKPOINT_ARG="--save_checkpoint"
fi

WANDB_FLAG=""
if [ "$USE_WANDB" = "true" ]; then
    WANDB_FLAG="--use_wandb"
fi

DETERMINISTIC_FLAG=""
if [ "$DETERMINISTIC" = "true" ]; then
    DETERMINISTIC_FLAG="--deterministic"
fi

# Reflow uses the same ZeRO-3 configuration as the baseline plus the reflow block.
if [ "$MODE" = "reflow" ]; then
cat > "$DS_CONFIG_JSON" << EOF
{
    "train_batch_size": $BATCH_SIZE,
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 0.0,
    "bf16": { "enabled": true },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": false,
        "reduce_bucket_size": 4e8,
        "sub_group_size": 1e9,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "reflow": {
            "enable_cpu_affinity": true,
            "bucketwise_cores_per_worker": ${WORKER_CORES:-8},
            "main_thread_cores": ${MAIN_CORES:-2}
        }
    },
    "wall_clock_breakdown": true
}
EOF

elif [ "$MODE" = "zerooffload" ]; then
cat > "$DS_CONFIG_JSON" << EOF
{
    "train_batch_size": $BATCH_SIZE,
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 0.0,
    "bf16": { "enabled": true },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": false,
        "reduce_bucket_size": 4e8,
        "sub_group_size": 1e9,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    },
    "wall_clock_breakdown": true
}
EOF
fi


CMD=(
deepspeed --num_gpus=$GPUS_PER_NODE --bind_cores_to_rank "$SCRIPT_DIR/finetune_zero3.py"
    --deepspeed_config="$DS_CONFIG_JSON"
    --model_name "$MODEL_NAME"
    --attn_implementation "$ATTN_IMPLEMENTATION"
    --num_train_epochs "$EPOCHS"
    --lr "$LR"
    --batch_size "$MICRO_BATCH_SIZE"
    --weight_decay "$WEIGHT_DECAY"
    --output_dir "$OUTPUT_DIR"
    --seed "$SEED"
    --max_length "$MAX_LENGTH"
    --log_interval "$LOG_INTERVAL"
    --dataset_name "$DATASET_NAME"
    --dataset_percentage "$DATASET_PERCENTAGE"
    --bench_steps "$BENCH_STEPS"
    --warmup_steps "$WARMUP_STEPS"
    $ACTIVATION_CHECKPOINTING_FLAG
    $SAVE_CHECKPOINT_ARG
    $WANDB_FLAG
    --wandb_project "$WANDB_PROJECT"
    --wandb_run_name "$WANDB_RUN_NAME"
    $DETERMINISTIC_FLAG
)

echo "Starting training with MODE $MODE"
echo "================================================"
"${CMD[@]}"

echo "================================================"
echo "Training completed"
echo "================================================"
