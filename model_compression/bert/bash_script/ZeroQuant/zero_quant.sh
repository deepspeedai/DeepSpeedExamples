#!/bin/bash

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%. run_jobs.sh (for mnli)


TASK_NAME=mnli  #mnli sst2 stsb mnli qqp rte cola mrpc qnli
export CUDA_VISIBLE_DEVICES=0
QGROUP=48
EPOCH=0
CONFIG=./config/ZeroQuant/ds_config_W8A8_Qgroup48_fp32.json
SAVE_PATH=./out/ZeroQuant/W8A8_quantization
mkdir -p ${SAVE_PATH}
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% if users provide *NO* models, use the following script %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% the following command will first download huggingface models and then compress %%%%%%%
MODEL=yoshitomo-matsubara/bert-base-uncased-${TASK_NAME} ## for both student and teacher
python -m torch.distributed.launch --nproc_per_node=1 \
  --master_port 66664 \
  run_glue_no_trainer.py \
  --seed 42 \
  --distill_method one_stage \
  --model_name_or_path ${MODEL} \
  --task_name $TASK_NAME \
  --max_length 128 \
  --pad_to_max_length \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 128 \
  --num_train_epochs ${EPOCH}\
  --eval_step 1000 \
  --deepspeed_config ${CONFIG} \
  --deepspeed \
  --save_best_model --clean_best_model \
  --gradient_accumulation_steps 1 \
  --output_dir ${SAVE_PATH} &>> ${SAVE_PATH}/train.log

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% users provide models  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MODEL_BASE=/blob/users/xwu/compression/huggingface_models/bert_base_uncased ## or you could use bert-base-uncased
# TEACHER=/blob/users/xwu/compression/huggingface_models/bert-base-uncased-${TASK_NAME}/pytorch_model.bin
# STUDENT=${TEACHER}
# python -m torch.distributed.launch --nproc_per_node=1 \
#   --master_port 66667 \
#   run_glue_no_trainer_clean.py \
#   --seed 42 \
#   --distill_method ${STAGE} \
#   --model_name_or_path ${MODEL_BASE} \
#   --pretrained_dir_student ${STUDENT} \
#   --pretrained_dir_teacher ${TEACHER} \
#   --task_name $TASK_NAME \
#   --max_length 128 \
#   --pad_to_max_length \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 18 \
#   --num_warmup_epochs 1 \
#   --deepspeed_config ${CONFIG} --weight_bit 1 \
#   --deepspeed \
#   --save_best_model --clean_best_model \
#   --gradient_accumulation_steps 1 \
#   --output_dir ${SAVE_PATH} &>> ${SAVE_PATH}/train.log
