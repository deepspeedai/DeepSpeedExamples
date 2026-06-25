# Default to a public HF model for out-of-the-box runs.
weight_path=facebook/opt-125m
export WANDB_MODE=disabled
num_gpus=${NUM_GPUS:-8}
epoch=3
mbs=2
MODE=${1:-zero2tp} 
if [ "$MODE" == "zero1tp" ]; then
  ZERO_STAGE=1
  AUTOTP_SIZE=4
  per_device_train_batch_size=$((mbs * AUTOTP_SIZE))
elif [ "$MODE" == "zero2tp" ]; then
  ZERO_STAGE=2
  AUTOTP_SIZE=4
  per_device_train_batch_size=$((mbs * AUTOTP_SIZE))
elif [ "$MODE" == "zero1" ]; then
  ZERO_STAGE=1
  AUTOTP_SIZE=0
  per_device_train_batch_size=$mbs
elif [ "$MODE" == "zero2" ]; then
  ZERO_STAGE=2
  AUTOTP_SIZE=0
  per_device_train_batch_size=$mbs
elif [ "$MODE" == "zero3" ]; then
  ZERO_STAGE=3
  AUTOTP_SIZE=0
  per_device_train_batch_size=$mbs
elif [ "$MODE" == "tp" ]; then
  ZERO_STAGE=0
  AUTOTP_SIZE=8
  per_device_train_batch_size=$((mbs * AUTOTP_SIZE))
else
  echo "error '$MODE',please use 'zero' or 'tp'。"
  exit 1
fi

# HF Trainer + Accelerate currently builds a 1D device mesh of size AUTOTP_SIZE.
# If num_gpus > AUTOTP_SIZE, ranks outside the mesh fail during init_device_mesh.
if [ "$AUTOTP_SIZE" -gt 1 ] && [ "$num_gpus" -ne "$AUTOTP_SIZE" ]; then
  echo "Adjusting num_gpus to AUTOTP_SIZE=$AUTOTP_SIZE to avoid device_mesh init failure."
  num_gpus=$AUTOTP_SIZE
fi
TEMPLATE_FILE="configs/ds_config_temp.json"
OUTPUT_FILE="configs/ds_config.json"
sed -e "s/\${zero_stage}/${ZERO_STAGE}/g" \
    -e "s/\${autotp_size}/${AUTOTP_SIZE}/g" \
    $TEMPLATE_FILE > $OUTPUT_FILE


deepspeed --num_gpus $num_gpus  \
    --master_port 51336  train.py  \
    --model_name_or_path  $weight_path \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir out_load_test/$MODE \
    --num_train_epochs $epoch \
    --gradient_checkpointing false \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 1 \
    --eval_strategy no \
    --save_strategy steps  \
    --save_steps 10000 \
    --gradient_accumulation_steps 4 \
    --learning_rate 0 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_steps 0 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --deepspeed "./configs/ds_config.json"
