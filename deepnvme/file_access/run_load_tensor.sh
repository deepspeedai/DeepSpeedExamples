#!/bin/bash

if [[ $# -ne 1 ]]; then 
    echo "Usage: $0 <input file on nvme device>"
    exit 1 
fi 

input_file=$1 
if ! [[ -f "$input_file" ]]; then
    echo "Error: $input_file does not exist"
    exit 1 
fi 


echo "Running load tensor examples using $input_file"
for f in aio_load_cpu_tensor.py aio_load_gpu_tensor.py \
    gds_load_gpu_tensor.py \
    py_load_cpu_tensor.py py_load_gpu_tensor.py; do 
    cmd="python $f --input_file $input_file"
    sync 
    echo $cmd 
    eval $cmd 
    sleep 2
done 


