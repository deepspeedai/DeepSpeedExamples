#!/bin/sh
export USE_TF=0 
BASE_LOG_DIR=~/experiments/zero_inference/
MODEL_NAME="opt-30b"
FULL_MODEL_NAME="facebook/${MODEL_NAME}"
QB=4

# zero-inference
BSZ=24
LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
mkdir -p  $LOG_DIR
deepspeed --num_gpus 1 run_model.py --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 1 &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_cpu_pin.txt 
deepspeed --num_gpus 1 run_model.py --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 1 --quant_bit 4 &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_cpu_pin_q${QB}.txt 

BSZ=96
LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
mkdir -p  $LOG_DIR
deepspeed --num_gpus 1 run_model.py --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 0 --kv-offload &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_cpu_kv.txt


BSZ=128
LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
mkdir -p  $LOG_DIR
deepspeed --num_gpus 1 run_model.py --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 0 --quant_bit ${QB} --kv-offload &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_cpu_q${QB}_kv.txt



# flexgen
BSZ=48
LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
mkdir -p  $LOG_DIR
python -m flexgen.flex_opt --model ${FULL_MODEL_NAME} --path _DUMMY_ --percent 0 100 100 0 100 0 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 &> $LOG_DIR/fg_${MODEL_NAME}_bs${BSZ}_cpu.txt    
python -m flexgen.flex_opt --model ${FULL_MODEL_NAME} --path _DUMMY_ --percent 0 100 100 0 100 0 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 --compress-weight  &> $LOG_DIR/fg_${MODEL_NAME}_bs${BSZ}_cpu_q4.txt    

BSZ=200
LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
mkdir -p  $LOG_DIR
python -m flexgen.flex_opt --model facebook/${MSZ} --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu.txt

BSZ=280
LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
mkdir -p  $LOG_DIR
python -m flexgen.flex_opt --model ${FULL_MODEL_NAME} --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 --compress-weight  &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_q4.txt    


