#!/bin/bash

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%. run_jobs.sh (for mnli)

export CUDA_VISIBLE_DEVICES=0
TASK_NAME=mnli  #mnli sst2 stsb mnli qqp rte cola mrpc qnli
STAGE=one_stage
LRATE=5e-5
QGROUP=1
EPOCH=18
WARMUP_EPOCH=1
#CONFIG=./config/ds_config_W1A8_Qgroup64_fp16.json # <=====================it's less stable
#CONFIG=./config/ds_config_W1or2A8_Qgroup64_fp16.json
CONFIG=./config/XTC/ds_config_layer_reduction_W1Q8_fp32.json
SAVE_PATH=./out/XTC/layer_reduction_W1A8_quantization
mkdir -p ${SAVE_PATH}
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% if users provide *NO* models, use the following script %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% the following command will first download huggingface models and then compress %%%%%%%
MODEL=yoshitomo-matsubara/bert-base-uncased-${TASK_NAME} ## for both student and teacher
student_model=./out/XTC/layer_reduction/best/pytorch_model.bin # <====================================================================Need student model
python -m torch.distributed.launch --nproc_per_node=1 \
  --master_port 66664 \
  run_glue_no_trainer.py \
  --pretrained_dir_student  ${student_model} \
  --seed 42 \
  --distill_method ${STAGE} \
  --model_name_or_path ${MODEL} \
  --task_name $TASK_NAME \
  --max_length 128 \
  --pad_to_max_length \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 128 \
  --learning_rate $LRATE \
  --num_train_epochs ${EPOCH}\
  --num_warmup_epochs ${WARMUP_EPOCH}  \
  --eval_step 1000 \
  --deepspeed_config ${CONFIG} \
  --deepspeed \
  --save_best_model --clean_best_model \
  --gradient_accumulation_steps 1 \
  --output_dir ${SAVE_PATH} &>> ${SAVE_PATH}/train.log
