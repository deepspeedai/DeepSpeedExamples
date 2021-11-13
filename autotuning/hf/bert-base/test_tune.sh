TASK_NAME=mnli
MODEL_NAME=bert-base-cased
HF_PATH=~/projects
PER_DEVICE_TRAIN_BATCH_SIZE=64
MAX_TRAIN_BATCH_SIZE=4096
NEPOCHS=1
NGPUS=16
NNODES=1
MAX_STEPS=200
OUTPUT_DIR=./${TASK_NAME}/output_b${PER_DEVICE_TRAIN_BATCH_SIZE}_g${NGPUS}_$MAX_STEPS

TEST=$1

if [ ${TEST} == "0" ]
then
    python -m torch.distributed.launch --nproc_per_node=$NGPUS $HF_PATH/transformers/examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --max_seq_length 128 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate 2e-5 \
    --num_train_epochs $NEPOCHS \
    --output_dir ${OUTPUT_DIR}_b${PER_DEVICE_TRAIN_BATCH_SIZE}_0 \
    --overwrite_output_dir \
    --save_steps 0 \
    --max_steps $MAX_STEPS \
    --save_strategy "no"
elif [ ${TEST} == "z0" ]
then
    deepspeed --num_nodes=$NNODES $HF_PATH/transformers/examples/pytorch/text-classification/run_glue.py --deepspeed ../dsconfigs/ds_config_z0.json \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --max_seq_length 128 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate 2e-5 \
    --num_train_epochs $NEPOCHS \
    --output_dir ${OUTPUT_DIR}_b${PER_DEVICE_TRAIN_BATCH_SIZE}_z0 \
    --save_steps 0 \
    --overwrite_output_dir \
    --max_steps $MAX_STEPS
elif [ ${TEST} == "z1" ]
then
    deepspeed --num_nodes=$NNODES $HF_PATH/transformers/examples/pytorch/text-classification/run_glue.py --deepspeed ../dsconfigs/ds_config_z1.json \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --max_seq_length 128 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate 2e-5 \
    --num_train_epochs $NEPOCHS \
    --output_dir ${OUTPUT_DIR}_b${PER_DEVICE_TRAIN_BATCH_SIZE}_z1 \
    --save_steps 0 \
    --overwrite_output_dir \
    --max_steps $MAX_STEPS
elif [ ${TEST} == "z2" ]
then
    deepspeed --num_nodes=$NNODES $HF_PATH/transformers/examples/pytorch/text-classification/run_glue.py --deepspeed ../dsconfigs/ds_config_z2.json \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --max_seq_length 128 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate 2e-5 \
    --num_train_epochs $NEPOCHS \
    --output_dir ${OUTPUT_DIR}_b${PER_DEVICE_TRAIN_BATCH_SIZE}_z2 \
    --save_steps 0 \
    --overwrite_output_dir \
    --max_steps $MAX_STEPS
elif [ ${TEST} == "z3" ]
then
    deepspeed --num_nodes=$NNODES $HF_PATH/transformers/examples/pytorch/text-classification/run_glue.py --deepspeed ../dsconfigs/ds_config_z3.json \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --max_seq_length 128 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate 2e-5 \
    --num_train_epochs $NEPOCHS \
    --output_dir ${OUTPUT_DIR}_b${PER_DEVICE_TRAIN_BATCH_SIZE}_z3 \
    --save_steps 0 \
    --overwrite_output_dir \
    --max_steps $MAX_STEPS
elif [ ${TEST} == "tune" ]
then
    deepspeed --autotuning run --num_nodes=$NNODES $HF_PATH/transformers/examples/pytorch/text-classification/run_glue.py --deepspeed ./ds_config_tune.json \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --max_seq_length 128 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate 2e-5 \
    --num_train_epochs $NEPOCHS \
    --output_dir ${OUTPUT_DIR}_tune \
    --save_steps 0 \
    --overwrite_output_dir \
    --max_steps $MAX_STEPS
elif [ ${TEST} == "fs" ]
then
    python -m torch.distributed.launch --nproc_per_node=$NGPUS $HF_PATH/transformers/examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --max_seq_length 128 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate 2e-5 \
    --num_train_epochs $NEPOCHS \
    --output_dir ${OUTPUT_DIR}_b${PER_DEVICE_TRAIN_BATCH_SIZE}_fs \
    --overwrite_output_dir \
    --save_steps 0 \
    --max_steps $MAX_STEPS \
    --sharded_ddp zero_dp_2
fi
