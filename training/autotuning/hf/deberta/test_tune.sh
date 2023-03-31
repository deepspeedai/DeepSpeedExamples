MODEL_NAME=microsoft/deberta-v2-xxlarge
TASK_NAME=mnli
PER_DEVICE_TRAIN_BATCH_SIZE=1
HF_PATH=~/projects
NEPOCHS=1
NGPUS=16
NNODES=1
MAX_STEPS=200
OUTPUT_DIR=./output_b${PER_DEVICE_TRAIN_BATCH_SIZE}_g${NGPUS}_$MAX_STEPS

TEST=$1

if [ ${TEST} == "0" ]
then
    python -m torch.distributed.launch --nproc_per_node=$NGPUS $HF_PATH/transformers/examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --fp16 \
    --max_seq_length 256 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate 3e-6 \
    --num_train_epochs $NEPOCHS \
    --output_dir ${OUTPUT_DIR}_0 \
    --overwrite_output_dir \
    --save_steps 0 \
    --max_steps $MAX_STEPS \
    --save_strategy "no"
elif [ ${TEST} == "z0" ]
then
    deepspeed --num_nodes=$NNODES --num_gpus=$NGPUS $HF_PATH/transformers/examples/pytorch/text-classification/run_glue.py --deepspeed ../dsconfigs/ds_config_fp16_z0.json\
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --fp16 \
    --max_seq_length 256 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate 3e-6 \
    --num_train_epochs $NEPOCHS \
    --output_dir ${OUTPUT_DIR}_z0 \
    --overwrite_output_dir \
    --save_steps 0 \
    --save_steps 0 \
    --max_steps $MAX_STEPS \
    --save_strategy "no"
elif [ ${TEST} == "z1" ]
then
    deepspeed --num_nodes=$NNODES --num_gpus=$NGPUS $HF_PATH/transformers/examples/pytorch/text-classification/run_glue.py --deepspeed ../dsconfigs/ds_config_fp16_z1.json\
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --fp16 \
    --max_seq_length 256 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate 3e-6 \
    --num_train_epochs $NEPOCHS \
    --output_dir ${OUTPUT_DIR}_z1 \
    --overwrite_output_dir \
    --save_steps 0 \
    --max_steps $MAX_STEPS \
    --save_strategy "no"
elif [ ${TEST} == "z2" ]
then
    deepspeed --num_nodes=$NNODES --num_gpus=$NGPUS $HF_PATH/transformers/examples/pytorch/text-classification/run_glue.py --deepspeed ../dsconfigs/ds_config_fp16_z2.json\
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --fp16 \
    --max_seq_length 256 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate 3e-6 \
    --num_train_epochs $NEPOCHS \
    --output_dir ${OUTPUT_DIR}_z2 \
    --overwrite_output_dir \
    --save_steps 0 \
    --max_steps $MAX_STEPS \
    --save_strategy "no"
elif [ ${TEST} == "z3" ]
then
    deepspeed --num_nodes=$NNODES --num_gpus=$NGPUS $HF_PATH/transformers/examples/pytorch/text-classification/run_glue.py --deepspeed ../dsconfigs/ds_config_fp16_z3.json\
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --fp16 \
    --max_seq_length 256 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate 3e-6 \
    --num_train_epochs $NEPOCHS \
    --output_dir ${OUTPUT_DIR}_z3 \
    --overwrite_output_dir \
    --save_steps 0 \
    --max_steps $MAX_STEPS \
    --save_strategy "no"
elif [ ${TEST} == "tune" ]
then
    deepspeed --autotuning run --num_nodes=$NNODES --num_gpus=$NGPUS $HF_PATH/transformers/examples/pytorch/text-classification/run_glue.py --deepspeed ./ds_config_fp16_tune.json\
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --fp16 \
    --max_seq_length 256 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate 3e-6 \
    --num_train_epochs $NEPOCHS \
    --output_dir ${OUTPUT_DIR}_tune \
    --overwrite_output_dir \
    --save_steps 0 \
    --max_steps $MAX_STEPS \
    --save_strategy "no"
elif [ ${TEST} == "fs" ]
then
    python -m torch.distributed.launch --nproc_per_node=$NGPUS $HF_PATH/transformers/examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --fp16 \
    --max_seq_length 256 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate 3e-6 \
    --num_train_epochs $NEPOCHS \
    --output_dir ${OUTPUT_DIR}_fs \
    --overwrite_output_dir \
    --save_steps 0 \
    --max_steps $MAX_STEPS \
    --save_strategy "no"
    --sharded_ddp zero_dp_2
fi
