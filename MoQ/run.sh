OOO=output
MASTER_PORT=12345
GPU=0

for TSK in qnli #stsb mrpc cola wnli sst2 rte qnli qqp mnli
do

if [ $TSK == wnli ] || [ $TSK == mrpc ]
then
    EPOCH_NUM=5
else
    EPOCH_NUM=3
fi

if [ $TSK == qqp ] || [ $TSK == mnli ]
then
    TEST_JSON=test_long.json
else
    TEST_JSON=test.json
fi

PORT=$((MASTER_PORT+GPU))

rm -rvf ./$OOO/${TSK}

CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch \
  --master_port $PORT \
  --nproc_per_node 1 run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TSK \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs $EPOCH_NUM \
  --output_dir ./$OOO/$TSK/ \
  --fp16 \
  --warmup_steps 2 \
  --deepspeed test.json

done
