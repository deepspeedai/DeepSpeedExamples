calc() { awk "BEGIN{ printf \"%.0f\n\", $* }"; }
dataset_name="ptb_text_only"
dataset_config_name="penn_treebank"
model_name_or_path="gpt2-medium"
global_batch_size=4
learning_rate=10e-5
warmup_ratio=0.0
decay_style="linear"
num_train_epochs=5
nproc_per_node=1

# baseline
seed=1234
cuda_device=0
master_port=$(( 12345 + ${cuda_device} ))
bash ds_finetune_gpt2.sh ${cuda_device} ${master_port} ${dataset_name} \
    ${dataset_config_name} ${model_name_or_path} ${global_batch_size} \
    ${learning_rate} ${warmup_ratio} ${decay_style} ${num_train_epochs} \
    ${seed} ${nproc_per_node} &


# random-LTD
ltd_enabled="true"
ltd_start=128
ltd_step_ratio=3e-1
step_per_epoch=262
ltd_step=$(calc $step_per_epoch*$num_train_epochs*$ltd_step_ratio)

seed=1234
cuda_device=1
master_port=$(( 12345 + ${cuda_device} ))
bash ds_finetune_gpt2.sh ${cuda_device} ${master_port} ${dataset_name} \
    ${dataset_config_name} ${model_name_or_path} ${global_batch_size} \
    ${learning_rate} ${warmup_ratio} ${decay_style} ${num_train_epochs} \
    ${seed} ${nproc_per_node} ${ltd_enabled} ${ltd_start} ${ltd_step} &


# curriculum learning
calc() { awk "BEGIN{ printf \"%.0f\n\", $* }"; }
dataset_name="ptb_text_only"
dataset_config_name="penn_treebank"
model_name_or_path="gpt2-medium"
global_batch_size=4
learning_rate=10e-5
warmup_ratio=0.0
decay_style="linear"
num_train_epochs=5
nproc_per_node=1

ltd_enabled="false"
ltd_start=128
ltd_step_ratio=3e-1
step_per_epoch=262
ltd_step=$(calc $step_per_epoch*$num_train_epochs*$ltd_step_ratio)
cl_enabled="true"
cl_num_metric=1
cl_1st_metric="seqlen_reshape"
cl_1st_index_to_sample_path="dummy"
cl_1st_index_to_metric_path="dummy"
cl_1st_difficulty_type="value"
cl_1st_clustering_type="single_cluster"
cl_1st_max=1024
cl_1st_difficulty_step=8
cl_1st_root=1
cl_1st_min=32
cl_1st_total_step=917

seed=1234
cuda_device=2
master_port=$(( 12345 + ${cuda_device} ))
bash ds_finetune_gpt2.sh ${cuda_device} ${master_port} ${dataset_name} \
    ${dataset_config_name} ${model_name_or_path} ${global_batch_size} \
    ${learning_rate} ${warmup_ratio} ${decay_style} ${num_train_epochs} \
    ${seed} ${nproc_per_node} ${ltd_enabled} ${ltd_start} ${ltd_step} \
    ${cl_enabled} ${cl_num_metric} ${cl_1st_metric} \
    ${cl_1st_index_to_sample_path} ${cl_1st_index_to_metric_path} \
    ${cl_1st_difficulty_type} ${cl_1st_clustering_type} ${cl_1st_min} \
    ${cl_1st_max} ${cl_1st_total_step} ${cl_1st_difficulty_step} \
    ${cl_1st_root} &

# curriculum learning + random-LTD
calc() { awk "BEGIN{ printf \"%.0f\n\", $* }"; }
dataset_name="ptb_text_only"
dataset_config_name="penn_treebank"
model_name_or_path="gpt2-medium"
global_batch_size=4
learning_rate=10e-5
warmup_ratio=0.0
decay_style="linear"
num_train_epochs=5
nproc_per_node=1

ltd_enabled="true"
ltd_start=128
cl_enabled="true"
cl_num_metric=1
cl_1st_metric="seqlen_reshape"
cl_1st_index_to_sample_path="dummy"
cl_1st_index_to_metric_path="dummy"
cl_1st_difficulty_type="value"
cl_1st_clustering_type="single_cluster"
cl_1st_max=1024
cl_1st_difficulty_step=8
cl_1st_root=1
cl_1st_min=32


cl_1st_total_step=131
ltd_step=393

seed=1234
cuda_device=3
master_port=$(( 12345 + ${cuda_device} ))
bash ds_finetune_gpt2.sh ${cuda_device} ${master_port} ${dataset_name} \
    ${dataset_config_name} ${model_name_or_path} ${global_batch_size} \
    ${learning_rate} ${warmup_ratio} ${decay_style} ${num_train_epochs} \
    ${seed} ${nproc_per_node} ${ltd_enabled} ${ltd_start} ${ltd_step} \
    ${cl_enabled} ${cl_num_metric} ${cl_1st_metric} \
    ${cl_1st_index_to_sample_path} ${cl_1st_index_to_metric_path} \
    ${cl_1st_difficulty_type} ${cl_1st_clustering_type} ${cl_1st_min} \
    ${cl_1st_max} ${cl_1st_total_step} ${cl_1st_difficulty_step} \
    ${cl_1st_root} &