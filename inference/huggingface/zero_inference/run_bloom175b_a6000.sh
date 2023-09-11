export USE_TF=0 
BASE_LOG_DIR=~/experiments/zero_inference/
MODEL_NAME="bloom"
FULL_MODEL_NAME="bigscience/${MODEL_NAME}"

OFFLOAD_DIR=/local_nvme/zero_offload
mkdir -p $OFFLOAD_DIR

QB=4

BSZ=8
LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
mkdir -p  $LOG_DIR
deepspeed --num_gpus 1 run_model.py --dummy --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --disk-offload --gen-len 32 --pin-memory 0 --offload-dir ${OFFLOAD_DIR} &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_disk.txt 

BSZ=4
LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
mkdir -p  $LOG_DIR
deepspeed --num_gpus 1 run_model.py --dummy --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 0 --offload-dir ${OFFLOAD_DIR} --quant_bits ${QB} &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_cpu_q${QB}.txt


BSZ=32
LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
mkdir -p  $LOG_DIR
deepspeed --num_gpus 1 run_model.py --dummy --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --disk-offload --gen-len 32 --pin-memory 0 --offload-dir ${OFFLOAD_DIR} --kv-offload &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_disk_kv.txt


BSZ=24
LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
mkdir -p  $LOG_DIR
deepspeed --num_gpus 1 run_model.py --dummy --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 0 --offload-dir ${OFFLOAD_DIR} --quant_bits ${QB} --kv-offload &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_cpu_q${QB}_kv.txt

