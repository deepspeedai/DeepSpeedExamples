MODEL_NAME=gpt2
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
    python -m torch.distributed.launch --nproc_per_node=$NGPUS $HF_PATH/transformers/examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --fp16 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate 2e-5 \
    --num_train_epochs $NEPOCHS \
    --output_dir ${OUTPUT_DIR}_0 \
    --overwrite_output_dir \
    --save_steps 0 \
    --max_steps $MAX_STEPS \
    --save_strategy "no"
elif [ ${TEST} == "z0" ]
then
    deepspeed --num_nodes=$NNODES --num_gpus=$NGPUS $HF_PATH/transformers/examples/pytorch/language-modeling/run_clm.py --deepspeed ../dsconfigs/ds_config_fp16_z0.json\
    --model_name_or_path $MODEL_NAME \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --fp16 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate 2e-5 \
    --num_train_epochs $NEPOCHS \
    --output_dir ${OUTPUT_DIR}_z0 \
    --overwrite_output_dir \
    --save_steps 0 \
    --max_steps $MAX_STEPS \
    --save_strategy "no"
elif [ ${TEST} == "z1" ]
then
    deepspeed --num_nodes=$NNODES --num_gpus=$NGPUS $HF_PATH/transformers/examples/pytorch/language-modeling/run_clm.py --deepspeed ../dsconfigs/ds_config_fp16_z1.json\
    --model_name_or_path $MODEL_NAME \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --fp16 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate 2e-5 \
    --num_train_epochs $NEPOCHS \
    --output_dir ${OUTPUT_DIR}_z1 \
    --overwrite_output_dir \
    --save_steps 0 \
    --max_steps $MAX_STEPS \
    --save_strategy "no"
elif [ ${TEST} == "z2" ]
then
    deepspeed --num_nodes=$NNODES --num_gpus=$NGPUS $HF_PATH/transformers/examples/pytorch/language-modeling/run_clm.py --deepspeed ../dsconfigs/ds_config_fp16_z2.json\
    --model_name_or_path $MODEL_NAME \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --fp16 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate 2e-5 \
    --num_train_epochs $NEPOCHS \
    --output_dir ${OUTPUT_DIR}_z2 \
    --overwrite_output_dir \
    --save_steps 0 \
    --max_steps $MAX_STEPS \
    --save_strategy "no"
elif [ ${TEST} == "z3" ]
then
    deepspeed --num_nodes=$NNODES --num_gpus=$NGPUS $HF_PATH/transformers/examples/pytorch/language-modeling/run_clm.py --deepspeed ../dsconfigs/ds_config_fp16_z3.json\
    --model_name_or_path $MODEL_NAME \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --fp16 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate 2e-5 \
    --num_train_epochs $NEPOCHS \
    --output_dir ${OUTPUT_DIR}_z3 \
    --overwrite_output_dir \
    --save_steps 0 \
    --max_steps $MAX_STEPS \
    --save_strategy "no"
elif [ ${TEST} == "tune" ]
then
    deepspeed --autotuning run --num_nodes=$NNODES --num_gpus=$NGPUS $HF_PATH/transformers/examples/pytorch/language-modeling/run_clm.py --deepspeed ../dsconfigs/ds_config_fp16_tune.json\
    --model_name_or_path $MODEL_NAME \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --fp16 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate 2e-5 \
    --num_train_epochs $NEPOCHS \
    --output_dir ${OUTPUT_DIR}_tune \
    --overwrite_output_dir \
    --save_steps 0 \
    --max_steps $MAX_STEPS \
    --save_strategy "no"
elif [ ${TEST} == "fs" ]
then
    python -m torch.distributed.launch --nproc_per_node=$NGPUS $HF_PATH/transformers/examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --fp16 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate 2e-5 \
    --num_train_epochs $NEPOCHS \
    --output_dir ${OUTPUT_DIR}_fs \
    --overwrite_output_dir \
    --save_steps 0 \
    --max_steps $MAX_STEPS \
    --save_strategy "no"
    --sharded_ddp zero_dp_2
fi
