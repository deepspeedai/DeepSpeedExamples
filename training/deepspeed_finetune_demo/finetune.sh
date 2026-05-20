NUM="${1:-2}"
MODEL="${2:-Qwen/Qwen2.5-0.5B}"
DATASET="${5:-tatsu-lab/alpaca}"
CONFIG="${3:-configs/z2_config.json}"
BATCH="${4:-8}"
deepspeed --num_gpus=$NUM --bind_cores_to_rank finetune_llama.py --model_name $MODEL --output_dir output --batch_size $BATCH --deepspeed_config $CONFIG --num_train_epochs 1 --eval_steps 100 --wandb_name $CONFIG-$BATCH --dataset_name $DATASET
