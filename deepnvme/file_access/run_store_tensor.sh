#!/bin/bash

if [[ $# -ne 1 ]]; then 
    echo "Usage: $0 <output folder on nvme device>"
    exit 1 
fi 

output_folder=$1 
if ! [[ -d "$output_folder" ]]; then
    echo "Error: $output_folder does not exist"
    exit 1 
fi 


echo "Running store tensor examples using $output_folder"
for f in aio_store_cpu_tensor.py aio_store_gpu_tensor.py \
    gds_store_gpu_tensor.py \
    py_store_cpu_tensor.py py_store_gpu_tensor.py; do 
    cmd="python $f --nvme_folder $output_folder"
    sync 
    echo $cmd 
    eval $cmd 
    sleep 2
done 


