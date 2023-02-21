#!/bin/bash

num_workers=1 # Num nodes to run the map job
num_threads=1 # Num threads on each node. Set this based on #CPU cores

# Which node is this node (start with 0 and end with num_workers-1). This
# script only launch the map job on 1 worker node, since we don't expect
# running on many nodes and workers don't need any communication. But you
# can modify this script to add a MPI/torch distributed launcher.
worker_id=$1
save_path="/blob/users/conglli/data/analysis_ptb_gpt/"

metric='total_vocab_freq'
# metric='vocab_rarity' # this requires the result of total_vocab_freq

dataset_name="ptb_text_only"
dataset_config_name="penn_treebank"
model_name_or_path="gpt2-medium"

batch_size=1000

jobname="gpt-ptb-analyzing-${metric}-map-worker${worker_id}"

options=" \
    --analyzing_task map \
    --analyzing_metric ${metric} \
    --analyzing_num_workers ${num_workers} \
    --analyzing_worker_id ${worker_id} \
    --analyzing_num_threads ${num_threads} \
    --dataset_name ${dataset_name} \
    --dataset_config_name ${dataset_config_name} \
    --model_name_or_path ${model_name_or_path} \
    --per_device_train_batch_size ${batch_size} \
    --output_dir ${save_path}"

python ../analyze_data.py ${options} &> ${jobname}.log