#!/bin/sh
export USE_TF=0 
BASE_LOG_DIR=~/experiments/zero_inference/
MODEL_NAME="opt-6.7b"
FULL_MODEL_NAME="facebook/${MODEL_NAME}"
BSZ=64
QB=4
LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
mkdir -p  $LOG_DIR


# deepspeed --num_gpus 1 run_model.py --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 1
deepspeed --num_gpus 1 run_model.py --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 1 --kv-offload

# deepspeed --num_gpus 1 run_model.py --model facebook/opt-350m --batch-size 1  --gen-len 32 #  --quant_bit 4
# deepspeed --num_gpus 1 run_model.py --model facebook/opt-350m --batch-size 1 --cpu-offload --gen-len 32 --pin-memory 0 --kv-offload  --quant_bit 4
# deepspeed --num_gpus 1 run_model.py --model facebook/${MSZ} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 0 --kv-offload &> $LOG_DIR/ds_${MSZ}_bs${BSZ}_cpu.txt 
# deepspeed --num_gpus 1 run_model.py --model facebook/${MSZ} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 1 --kv-offload &> $LOG_DIR/ds_${MSZ}_bs${BSZ}_cpu_pin.txt
# deepspeed --num_gpus 1 run_model.py --model facebook/${MSZ} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 1 --kv-offload --quant_bit 4 &> $LOG_DIR/ds_${MSZ}_bs${BSZ}_cpu_pin_q${QB}.txt

# # 1.3b flexgen with compute schedule or partial offload
# python -m flexgen.flex_opt --model facebook/${MSZ} --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu.txt
# python -m flexgen.flex_opt --model facebook/${MSZ} --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 1 --num-gpu-batches 1 &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_pin.txt

# mkdir -p  $LOG_DIR
# OFFLOAD_DIR=/local_nvme/flexgen_offload
# mkdir -p $OFFLOAD_DIR
# python -m flexgen.flex_opt --model facebook/${MSZ} --path _DUMMY_ --percent 0 0 0 100 0 100 --gpu-batch-size ${BSZ} --offload-dir ${OFFLOAD_DIR} --pin-weight 0 --num-gpu-batches 1 &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_disk.txt
# python -m flexgen.flex_opt --model facebook/${MSZ} --path _DUMMY_ --percent 0 0 0 100 0 100 --gpu-batch-size ${BSZ} --offload-dir ${OFFLOAD_DIR} --pin-weight 0 --num-gpu-batches 1 --cpu-cache-compute &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_ccc_disk.txt
