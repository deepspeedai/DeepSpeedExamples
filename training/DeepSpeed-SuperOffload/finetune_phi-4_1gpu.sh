#!/bin/bash
set -e

echo "================================================"
echo "Phi-4 Fine-tuning with DeepSpeed on 1 GPU"
echo "================================================"

# MODE=Options: "superoffload" or "zerooffload"
MODE=$1
BATCH_SIZE=${2:-4}

SCRIPT_DIR=$(dirname "$0")
MODEL_NAME="microsoft/phi-4"
OUTPUT_DIR="${SCRIPT_DIR}/phi-4_${MODE}_output"
DS_CONFIG_JSON="${SCRIPT_DIR}/phi-4_${MODE}_config.json"

mkdir -p $OUTPUT_DIR

# Script argument parameters
ACTIVATION_CHECKPOINTING=true
SAVE_CHECKPOINT=false
MAX_LENGTH=4096
LOG_INTERVAL=1
DATASET_NAME="tatsu-lab/alpaca"
DATASET_PERCENTAGE=10.0
USE_WANDB=false
WANDB_PROJECT="phi-4"
WANDB_RUN_NAME="phi-4-$MODE"
DETERMINISTIC=false
BENCH_STEPS=10
WARMUP_STEPS=20

EPOCHS=1
LR=1e-5
WARMUP=0.05
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

# Create DeepSpeed configuration file
if [ "$MODE" = "superoffload" ]; then
cat > "$DS_CONFIG_JSON" << EOF
{
    "train_batch_size": $BATCH_SIZE,
    "gradient_accumulation_steps": 1,
    "bf16": { "enabled": true },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": false,
        "reduce_bucket_size": 4e8,
        "sub_group_size": 4e8,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true,
            "ratio": 0.90,
            "super_offload": true,
            "cpuadam_cores_perc": 0.90
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
    "bf16": { "enabled": true },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": false,
        "reduce_bucket_size": 4e8,
        "sub_group_size": 4e8,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    },
    "wall_clock_breakdown": true
}
EOF
fi

GPUS_PER_NODE=1

CMD="deepspeed --num_gpus=$GPUS_PER_NODE finetune_zero3.py \
    --deepspeed_config=$DS_CONFIG_JSON \
    --model_name $MODEL_NAME \
    --num_train_epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --weight_decay $WEIGHT_DECAY \
    --output_dir $OUTPUT_DIR \
    --seed $SEED \
    --max_length $MAX_LENGTH \
    --log_interval $LOG_INTERVAL \
    --dataset_name $DATASET_NAME \
    --dataset_percentage $DATASET_PERCENTAGE \
    --bench_steps $BENCH_STEPS \
    --warmup_steps $WARMUP_STEPS \
    $ACTIVATION_CHECKPOINTING_FLAG \
    $SAVE_CHECKPOINT_ARG \
    $WANDB_FLAG \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME \
    $DETERMINISTIC_FLAG"

echo "Starting training with MODE $MODE"
echo "================================================"
eval $CMD
