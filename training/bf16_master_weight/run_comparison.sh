#!/bin/bash
# Run memory comparison across different bf16 configurations
#
# Usage:
#   ./run_comparison.sh [--num_gpus N] [--num_layers L] [--hidden_dim H]
#
# This script runs training with two configurations:
#   1. baseline - Standard bf16 with fp32 master weights/grads/optimizer states
#   2. bf16_full - bf16 master weights, gradients, and optimizer states

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default settings
NUM_GPUS=1
NUM_LAYERS=12
HIDDEN_DIM=1024
BATCH_SIZE=4
SEQ_LENGTH=512
NUM_STEPS=20
WARMUP_STEPS=5
MASTER_PORT=29600

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --num_layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        --hidden_dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --seq_length)
            SEQ_LENGTH="$2"
            shift 2
            ;;
        --num_steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create logs directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "BF16 Low-Precision Master Weights Memory Test"
echo "=============================================="
echo "Configuration:"
echo "  NUM_GPUS: $NUM_GPUS"
echo "  NUM_LAYERS: $NUM_LAYERS"
echo "  HIDDEN_DIM: $HIDDEN_DIM"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  SEQ_LENGTH: $SEQ_LENGTH"
echo "  NUM_STEPS: $NUM_STEPS"
echo "  LOG_DIR: $LOG_DIR"
echo "=============================================="

COMMON_ARGS="--num_layers $NUM_LAYERS --hidden_dim $HIDDEN_DIM --batch_size $BATCH_SIZE --seq_length $SEQ_LENGTH --num_steps $NUM_STEPS --warmup_steps $WARMUP_STEPS"

# Run baseline configuration
echo ""
echo "[1/2] Running BASELINE (bf16 with fp32 master weights/grads/optimizer states)..."
echo "----------------------------------------------"
deepspeed --num_gpus=$NUM_GPUS --master_port=$MASTER_PORT train.py \
    --deepspeed_config configs/baseline.json \
    $COMMON_ARGS \
    2>&1 | tee "$LOG_DIR/baseline.log"

# Run bf16_full configuration
echo ""
echo "[2/2] Running BF16_FULL (bf16 master weights, gradients, and optimizer states)..."
echo "----------------------------------------------"
deepspeed --num_gpus=$NUM_GPUS --master_port=$MASTER_PORT train.py \
    --deepspeed_config configs/bf16_full.json \
    $COMMON_ARGS \
    2>&1 | tee "$LOG_DIR/bf16_full.log"

echo ""
echo "=============================================="
echo "All runs complete. Gathering results..."
echo "=============================================="

# Run the gather script to create summary
python gather_memory.py --log_dir "$LOG_DIR"

echo ""
echo "Results saved to: $LOG_DIR/summary.txt"
echo "=============================================="
