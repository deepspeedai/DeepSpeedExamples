#!/bin/bash

# Set these 2 to the same as what you used during map job. We need these 2
# configs to know how many map job result files do we have.
num_workers=1
num_threads=1
# Reduce job only has 1 worker but can accelerate by multithreading.
num_threads_reduce=1

save_path="/blob/users/conglli/data/analysis_ptb_gpt/"

metric='total_vocab_freq'
# metric='vocab_rarity' # this requires the result of total_vocab_freq

dataset_name="ptb_text_only"
dataset_config_name="penn_treebank"
model_name_or_path="gpt2-medium"

batch_size=1000

jobname="gpt-ptb-analyzing-${metric}-reduce"

options=" \
    --analyzing_task reduce \
    --analyzing_metric ${metric} \
    --analyzing_num_workers ${num_workers} \
    --analyzing_num_threads ${num_threads} \
    --analyzing_num_threads_reduce ${num_threads_reduce} \
    --dataset_name ${dataset_name} \
    --dataset_config_name ${dataset_config_name} \
    --model_name_or_path ${model_name_or_path} \
    --per_device_train_batch_size ${batch_size} \
    --output_dir ${save_path}"

python ../analyze_data.py ${options} &> ${jobname}.log