#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
GPUS_PER_NODE=2
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

# Model parameters
MODEL_NAME="meta-llama/Llama-2-7b-hf"
OUTPUT_DIR="./alpaca_output"
EPOCHS=3
SEED=42

# ZenFlow config file path
DS_CONFIG_JSON="./zf_config.json"

# Note: LR, batch_size, weight_decay are defined in the config file
# These parameters are kept for fallback only
LR=2e-5
BATCH_SIZE=8
WARMUP=0.03
WEIGHT_DECAY=0.01

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# DeepSpeed command
if [ -f "$DS_CONFIG_JSON" ]; then
    echo "[INFO] Using DeepSpeed config file: $DS_CONFIG_JSON"
    CMD="deepspeed --num_gpus=$GPUS_PER_NODE finetune_llama.py \
        --deepspeed_config=$DS_CONFIG_JSON \
        --model_name $MODEL_NAME \
        --num_train_epochs $EPOCHS \
        --lr $LR \
        --batch_size $BATCH_SIZE \
        --weight_decay $WEIGHT_DECAY \
        --output_dir $OUTPUT_DIR \
        --seed $SEED"
else
    echo "[ERROR] DeepSpeed config file not found: $DS_CONFIG_JSON"
    exit 1
fi

echo "[INFO] Running DeepSpeed training with ZenFlow:"
echo $CMD
eval $CMD