#!/bin/bash

cuda_device=$1
master_port=$2
dataset_name=$3
dataset_config_name=$4
model_name_or_path=$5
global_batch_size=$6
learning_rate=$7
warmup_ratio=$8
decay_style=$9
num_train_epochs=${10}
seed=${11}
nproc_per_node=${12}

### Random layerwise token dropping (random-LTD) configs
## random-LTD's main switch. "false" means disabled. "true" means enabled.
ltd_enabled=${13:-'false'}
## How much dropping ratio to start with. The value denotes the seqlen after
## dropping.
ltd_start=${14:-1024}
## How many steps for random-LTD to gradually reduce dropping ratio to zero.
ltd_step=${15:-1}

cl_enabled=${16:-'false'}
## Number of CL metrics to use.
cl_num_metric=${17:-1}

## Name of difficulty metric
cl_1st_metric=${18:-'dummy'}
## Path to the data indexes for this difficulty metric. Samples on ith row of
## index_to_sample have the difficulty value equals to ith row of
## index_to_metric.
cl_1st_index_to_sample_path=${19:-'dummy'}
cl_1st_index_to_metric_path=${20:-'dummy'}
## During training, whether increase difficulty by value- or percentile-based.
cl_1st_difficulty_type=${21:-'value'}
## "single_cluster" means no clustering required and probably CL is achieved by
## data postprocessing. "schedule_based" means will cluster data based on the
## difficulty schedule (pacing function) below.
cl_1st_clustering_type=${22:-'single_cluster'}
## Start difficulty
cl_1st_min=${23:-2048}
## End difficulty
cl_1st_max=${24:-2048}
## Total step to reach end difficulty
cl_1st_total_step=${25:-1}
## When changing difficulty, always make sure it's a multiple of the
## difficulty_step below.
cl_1st_difficulty_step=${26:-1}
## Root degree of the schedule (pacing function).
cl_1st_root=${27:-1}

cl_2nd_metric=${28:-'dummy'}
cl_2nd_index_to_sample_path=${29:-'dummy'}
cl_2nd_index_to_metric_path=${30:-'dummy'}
cl_2nd_difficulty_type=${31:-'value'}
cl_2nd_clustering_type=${32:-'single_cluster'}
cl_2nd_min=${33:-2048}
cl_2nd_max=${34:-2048}
cl_2nd_total_step=${35:-1}
cl_2nd_difficulty_step=${36:-1}
cl_2nd_root=${37:-1}

micro_batch_size=2
gradient_accumulation_steps=$(( ${global_batch_size} / ${micro_batch_size} / ${nproc_per_node} ))

export CUDA_VISIBLE_DEVICES=${cuda_device}

jobname="${model_name_or_path}_${dataset_name}_${dataset_config_name}_gbs${global_batch_size}_mbs${micro_batch_size}"
jobname="${jobname}_lr${learning_rate}_warmup${warmup_ratio}_decay${decay_style}_epoch${num_train_epochs}_seed${seed}"
if [ "${ltd_enabled}" = "true" ]; then
    jobname="${jobname}_ltd_${ltd_start}_${ltd_step}"
fi
if [ "${cl_enabled}" = "true" ]; then
    jobname="${jobname}_cl_${cl_1st_metric}_${cl_1st_min}_${cl_1st_max}_${cl_1st_total_step}_${cl_1st_root}"
    if [[ $cl_num_metric -gt 1 ]]; then
        jobname="${jobname}_${cl_2nd_metric}_${cl_2nd_min}_${cl_2nd_max}_${cl_2nd_total_step}_${cl_2nd_root}"
    fi
    data_cluster_path="./data_cluster/${jobname}"
    mkdir -p ${data_cluster_path}
    num_workers=0
fi
config_json="ds_config_${model_name_or_path}_gbs${global_batch_size}_mbs${micro_batch_size}"
if [ "${ltd_enabled}" = "true" ]; then
    config_json="${config_json}_ltd_${ltd_start}_${ltd_step}"
fi
if [ "${cl_enabled}" = "true" ]; then
    config_json="${config_json}_cl_${cl_1st_metric}_${cl_1st_min}_${cl_1st_max}_${cl_1st_total_step}_${cl_1st_root}"
    if [[ $cl_num_metric -gt 1 ]]; then
        config_json="${config_json}_${cl_2nd_metric}_${cl_2nd_min}_${cl_2nd_max}_${cl_2nd_total_step}_${cl_2nd_root}"
    fi
fi
config_json="${config_json}.json"
if [[ $cl_num_metric -gt 1 ]]; then
template_json="ds_config_${model_name_or_path}_2clmetrics_TEMPLATE.json"
sed "s/LTD_ENABLED/${ltd_enabled}/" ${template_json} \
    | sed "s/LTD_MIN/${ltd_start}/" \
    | sed "s/LTD_STEP/${ltd_step}/" \
    | sed "s/GB_SIZE/${global_batch_size}/" \
    | sed "s/MB_SIZE/${micro_batch_size}/" \
    | sed "s/DATA_EFFICIENCY_SEED/${seed}/" \
    | sed "s/CL_ENABLED/${cl_enabled}/" \
    | sed "s/DATA_SAMPLING_NUM_WORKERS/${num_workers}/" \
    | sed "s#CL_CLUSTER_PATH#${data_cluster_path}#" \
    | sed "s#CL_1st_METRIC_NAME#${cl_1st_metric}#" \
    | sed "s#CL_1st_SAMPLE_PATH#${cl_1st_index_to_sample_path}#" \
    | sed "s#CL_1st_METRIC_PATH#${cl_1st_index_to_metric_path}#" \
    | sed "s#CL_1st_DIFF_TYPE#${cl_1st_difficulty_type}#" \
    | sed "s#CL_1st_CLUSTER_TYPE#${cl_1st_clustering_type}#" \
    | sed "s/CL_1st_MIN/${cl_1st_min}/" \
    | sed "s/CL_1st_MAX/${cl_1st_max}/" \
    | sed "s/CL_1st_TOTAL_STEP/${cl_1st_total_step}/" \
    | sed "s/CL_1st_DIFF_STEP/${cl_1st_difficulty_step}/" \
    | sed "s/CL_1st_ROOT/${cl_1st_root}/" \
    | sed "s#CL_2nd_METRIC_NAME#${cl_2nd_metric}#" \
    | sed "s#CL_2nd_SAMPLE_PATH#${cl_2nd_index_to_sample_path}#" \
    | sed "s#CL_2nd_METRIC_PATH#${cl_2nd_index_to_metric_path}#" \
    | sed "s#CL_2nd_DIFF_TYPE#${cl_2nd_difficulty_type}#" \
    | sed "s#CL_2nd_CLUSTER_TYPE#${cl_2nd_clustering_type}#" \
    | sed "s/CL_2nd_MIN/${cl_2nd_min}/" \
    | sed "s/CL_2nd_MAX/${cl_2nd_max}/" \
    | sed "s/CL_2nd_TOTAL_STEP/${cl_2nd_total_step}/" \
    | sed "s/CL_2nd_DIFF_STEP/${cl_2nd_difficulty_step}/" \
    | sed "s/CL_2nd_ROOT/${cl_2nd_root}/" \
    > ${config_json}
else
template_json="ds_config_${model_name_or_path}_1clmetric_TEMPLATE.json"
sed "s/LTD_ENABLED/${ltd_enabled}/" ${template_json} \
    | sed "s/LTD_MIN/${ltd_start}/" \
    | sed "s/LTD_STEP/${ltd_step}/" \
    | sed "s/GB_SIZE/${global_batch_size}/" \
    | sed "s/MB_SIZE/${micro_batch_size}/" \
    | sed "s/DATA_EFFICIENCY_SEED/${seed}/" \
    | sed "s/CL_ENABLED/${cl_enabled}/" \
    | sed "s/DATA_SAMPLING_NUM_WORKERS/${num_workers}/" \
    | sed "s#CL_CLUSTER_PATH#${data_cluster_path}#" \
    | sed "s#CL_1st_METRIC_NAME#${cl_1st_metric}#" \
    | sed "s#CL_1st_SAMPLE_PATH#${cl_1st_index_to_sample_path}#" \
    | sed "s#CL_1st_METRIC_PATH#${cl_1st_index_to_metric_path}#" \
    | sed "s#CL_1st_DIFF_TYPE#${cl_1st_difficulty_type}#" \
    | sed "s#CL_1st_CLUSTER_TYPE#${cl_1st_clustering_type}#" \
    | sed "s/CL_1st_MIN/${cl_1st_min}/" \
    | sed "s/CL_1st_MAX/${cl_1st_max}/" \
    | sed "s/CL_1st_TOTAL_STEP/${cl_1st_total_step}/" \
    | sed "s/CL_1st_DIFF_STEP/${cl_1st_difficulty_step}/" \
    | sed "s/CL_1st_ROOT/${cl_1st_root}/" \
    > ${config_json}
fi
# if you want to save the final checkpoint, add "--output_dir checkpoint_path" to options
options=" \
    --dataset_name ${dataset_name} \
    --dataset_config_name ${dataset_config_name} \
    --model_name_or_path ${model_name_or_path} \
    --per_device_train_batch_size ${micro_batch_size} \
    --per_device_eval_batch_size ${micro_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learning_rate} \
    --warmup_ratio ${warmup_ratio} \
    --decay_style ${decay_style} \
    --token_based_lr_decay \
    --num_train_epochs ${num_train_epochs}  \
    --seed ${seed} \
    --deepspeed_config ${config_json} \
    --deepspeed"
if [ "${ltd_enabled}" = "true" ]; then
options="${options} \
    --random_ltd"
fi
if [ "${cl_enabled}" = "true" ]; then
options="${options} \
    --curriculum_learning"
fi
mkdir -p ./output/${model_name_or_path}
python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} \
    --master_port ${master_port} ../run_clm_no_trainer.py \
    ${options} &> ./output/${model_name_or_path}/${jobname}.log
