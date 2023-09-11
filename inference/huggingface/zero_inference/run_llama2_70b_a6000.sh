#!/bin/sh
export USE_TF=0 
BASE_LOG_DIR=~/experiments/zero_inference/
MODEL_NAME="Llama-2-70b-hf"
FULL_MODEL_NAME="meta-llama/${MODEL_NAME}"
QB=4

BSZ=64
LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
mkdir -p  $LOG_DIR
deepspeed --num_gpus 1 run_model.py --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 1 &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_cpu_pin.txt 

BSZ=96
LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
mkdir -p  $LOG_DIR
deepspeed --num_gpus 1 run_model.py --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 1 --quant_bit ${QB} &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_cpu_pin_q${QB}.txt 
deepspeed --num_gpus 1 run_model.py --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 0 --kv-offload &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_cpu_kv.txt 


BSZ=200
LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
mkdir -p  $LOG_DIR
deepspeed --num_gpus 1 run_model.py --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 0 --kv-offload --quant_bit ${QB} &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_cpu_kv_q${QB}.txt
