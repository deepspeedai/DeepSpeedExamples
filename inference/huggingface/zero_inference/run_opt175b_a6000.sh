export USE_TF=0 
BASE_LOG_DIR=~/experiments/zero_inference/
MODEL_NAME="opt-175b"
FULL_MODEL_NAME="facebook/${MODEL_NAME}"
QB=4

OFFLOAD_DIR=/local_nvme/zero_offload
mkdir -p $OFFLOAD_DIR

# zero-inference
BSZ=8
LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
mkdir -p  $LOG_DIR

deepspeed --num_gpus 1 run_model.py --dummy --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --disk-offload --gen-len 32 --pin-memory 0 --offload-dir ${OFFLOAD_DIR} &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_disk.txt 
deepspeed --num_gpus 1 run_model.py --dummy --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 0 --offload-dir ${OFFLOAD_DIR} --quant_bits ${QB} &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_cpu_q${QB}.txt

BSZ=32
LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
mkdir -p  $LOG_DIR
deepspeed --num_gpus 1 run_model.py --dummy --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --disk-offload --gen-len 32 --pin-memory 0 --offload-dir ${OFFLOAD_DIR} --kv-offload &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_disk_kv.txt


BSZ=24
LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
mkdir -p  $LOG_DIR
deepspeed --num_gpus 1 run_model.py --dummy --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 0 --offload-dir ${OFFLOAD_DIR} --quant_bits ${QB} --kv-offload &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_cpu_q${QB}_kv.txt


# flexgen
OFFLOAD_DIR=/local_nvme/flexgen_offload
mkdir -p $OFFLOAD_DIR

BSZ=16
LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
mkdir -p  $LOG_DIR
python -m flexgen.flex_opt --model ${FULL_MODEL_NAME} --path _DUMMY_ --percent 0 0 100 0 100 0 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 &> $LOG_DIR/fg_${MODEL_NAME}_bs${BSZ}_disk.txt    
python -m flexgen.flex_opt --model ${FULL_MODEL_NAME} --path _DUMMY_ --percent 0 100 100 0 100 0 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 --compress-weight  &> $LOG_DIR/fg_${MODEL_NAME}_bs${BSZ}_cpu_q4.txt    

BSZ=64
LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
mkdir -p  $LOG_DIR
python -m flexgen.flex_opt --model ${FULL_MODEL_NAME} --path _DUMMY_ --percent 0 0 0 100 0 100 --gpu-batch-size ${BSZ} --offload-dir ${OFFLOAD_DIR} --pin-weight 0 --num-gpu-batches 1  &> $LOG_DIR/fg_${MODEL_NAME}_bs${BSZ}_cpu_disk.txt

BSZ=40
LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
mkdir -p  $LOG_DIR
python -m flexgen.flex_opt --model ${FULL_MODEL_NAME} --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 --compress-weight  &> $LOG_DIR/fg_${MODEL_NAME}_bs${BSZ}_cpu_q4.txt    

