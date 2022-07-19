#!/bin/bash
DIR=`pwd`
export CUDA_VISIBLE_DEVICES=2
TASK_NAME=mnli  #mnli sst2 stsb mnli qqp rte cola mrpc qnli
STAGE=one_stage
LRATE=5e-5
EPOCH=2
WARMUP_EPOCH=1
BATCH_SIZE_PER_GPU=32
NAME="quant_activation"
SAVE_PATH=./out/${NAME}
mkdir -p ${SAVE_PATH}

###Layer Reduction
LAYER_REDUCTION_ENABLE="false" 
FP16_ENABLE="false"

###weight quantization
WEIGHT_QUANT_ENABLE="false"
Q_GROUP=64
W_BIT1=4
W_BIT2=2
###activation quantization
ACTIVATION_QUANT_ENABLE="true" #<=============================================================
A_BIT1=8                       #<=============================================================
A_BIT2=4                       #<=============================================================
#############pruning
###sparse_pruning (unstructure pruning)
SPARSE_PRUNING_ENABLE="false"
S_DENSE_RATIO=0.6
###row_pruning (unstructure pruning)
ROW_PRUNING_ENABLE="false"
R_DENSE_RATIO=0.6
###HEAD_PRUNING_ENABLE
HEAD_PRUNING_ENABLE="false"
H_DENSE_RATIO=0.6

template_json="config/ds_config_TEMPLATE.json"
config_json="config/ds_config_${NAME}.json"

if [ "${FP16_ENABLE}" = "true" ]; then
    QuantW_FORWARD="false"
else
    QuantW_FORWARD="true"
fi
sed "s/LAYER_REDUCTION_ENABLE/${LAYER_REDUCTION_ENABLE}/" ${template_json} \
    | sed "s/WEIGHT_QUANT_ENABLE/${WEIGHT_QUANT_ENABLE}/" \
    | sed "s/Q_GROUP/${Q_GROUP}/" \
    | sed "s/W_BIT1/${W_BIT1}/" \
    | sed "s/W_BIT2/${W_BIT2}/" \
    | sed "s/ACTIVATION_QUANT_ENABLE/${ACTIVATION_QUANT_ENABLE}/" \
    | sed "s/A_BIT1/${A_BIT1}/" \
    | sed "s/A_BIT2/${A_BIT2}/" \
    | sed "s/SPARSE_PRUNING_ENABLE/${SPARSE_PRUNING_ENABLE}/" \
    | sed "s/S_DENSE_RATIO/${S_DENSE_RATIO}/" \
    | sed "s/ROW_PRUNING_ENABLE/${ROW_PRUNING_ENABLE}/" \
    | sed "s/R_DENSE_RATIO/${R_DENSE_RATIO}/" \
    | sed "s/HEAD_PRUNING_ENABLE/${HEAD_PRUNING_ENABLE}/" \
    | sed "s/H_DENSE_RATIO/${H_DENSE_RATIO}/" \
    | sed "s/FP16_ENABLE/${FP16_ENABLE}/" \
    | sed "s/QuantW_FORWARD/${QuantW_FORWARD}/" \
    | sed "s/BATCH_SIZE_PER_GPU/${BATCH_SIZE_PER_GPU}/" \
      > ${config_json}
      
CONFIG=${config_json}
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% if users provide *NO* models, use the following script %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% the following command will first download huggingface models and then compress %%%%%%%
MODEL=yoshitomo-matsubara/bert-base-uncased-${TASK_NAME} ## for both student and teacher
run_cmd="python -m torch.distributed.launch --nproc_per_node=1 \
  --master_port 66666 \
  run_glue_no_trainer.py \
  --seed 42 \
  --distill_method ${STAGE} \
  --model_name_or_path ${MODEL} \
  --task_name $TASK_NAME \
  --max_length 128 \
  --pad_to_max_length \
  --per_device_train_batch_size ${BATCH_SIZE_PER_GPU} \
  --per_device_eval_batch_size 64 \
  --learning_rate $LRATE \
  --num_train_epochs ${EPOCH}\
  --num_warmup_epochs ${WARMUP_EPOCH}  \
  --eval_step 1000 \
  --deepspeed_config ${CONFIG} \
  --deepspeed \
  --save_best_model --clean_best_model \
  --gradient_accumulation_steps 1 \
  --output_dir ${SAVE_PATH} | tee -a  ${SAVE_PATH}/train.log"

echo ${run_cmd}
eval ${run_cmd}
set +x