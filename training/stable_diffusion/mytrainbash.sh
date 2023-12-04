export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export OUTPUT_DIR="./sd-distill-v21"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi
    

accelerate launch train_sd_distil_lora.py \
	   --pretrained_model_name_or_path=$MODEL_NAME  \
	   --output_dir=$OUTPUT_DIR \
	   --default_prompt="A man dancing" \
	   --resolution=512 \
	   --train_batch_size=1 \
	   --gradient_accumulation_steps=1 \
	   --learning_rate=5e-6 \
	   --lr_scheduler="constant" \
	   --lr_warmup_steps=0
