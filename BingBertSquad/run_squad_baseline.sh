#~/bin/bash

#1: number of GPUs
#2: batch size per GPU
#3: gradient accumulation step
#4: --fp16 if you want to enable fp16
#5: --deepspeed if you want to enable deepspeed
NGPU=$1
EFFECTIVE_BATCH_SIZE=24
MAX_GPU_BATCH_SIZE=6
PER_GPU_BATCH_SIZE=$((EFFECTIVE_BATCH_SIZE/NGPU))
if [[ $PER_GPU_BATCH_SIZE -lt $MAX_GPU_BATCH_SIZE ]]; then
       GRAD_ACCUM_STEPS=1
else
       GRAD_ACCUM_STEPS=$((PER_GPU_BATCH_SIZE/MAX_GPU_BATCH_SIZE))
fi
LR=3e-5
MASTER_PORT=$((NGPU+12345))
JOB_NAME="baseline_${NGPU}GPUs_${EFFECTIVE_BATCH_SIZE}batch_size"
run_cmd="python3.6 -m torch.distributed.launch \
       --nproc_per_node=${NGPU} \
       --master_port=${MASTER_PORT} \
       nvidia_run_squad_baseline.py \
       --bert_model bert-large-uncased \
       --do_train \
       --do_lower_case \
       --do_predict \
       --train_file $SQUAD_DIR/train-v1.1.json \
       --predict_file $SQUAD_DIR/dev-v1.1.json \
       --train_batch_size $PER_GPU_BATCH_SIZE \
       --learning_rate ${LR} \
       --num_train_epochs 2.0 \
       --max_seq_length 384 \
       --doc_stride 128 \
       --output_dir $OUTPUT_DIR \
       --job_name ${JOB_NAME} \
       --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
       --fp16 \
       --model_file $2
       "
echo ${run_cmd}
eval ${run_cmd}

