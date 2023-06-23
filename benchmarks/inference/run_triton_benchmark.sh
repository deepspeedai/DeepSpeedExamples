date=`date +%Y%m%d`
trials=12

model=bert-base-cased
log_path=results/${date}_${model}
mkdir -p ${log_path}
deepspeed --num_gpus 1 triton-bert-benchmark.py --model ${model} --dtype fp16 --trials ${trials} --deepspeed --graphs &> ${log_path}/hf-graph.log
deepspeed --num_gpus 1 triton-bert-benchmark.py --model ${model} --dtype fp16 --trials ${trials} --deepspeed --kernel-inject --graphs &> ${log_path}/ds-graph.log
deepspeed --num_gpus 1 triton-bert-benchmark.py --model ${model} --dtype fp16 --trials ${trials} --deepspeed --kernel-inject --graphs --triton  &> ${log_path}/triton-graph.log


model=bert-large-cased
log_path=results/${date}_${model}
mkdir -p ${log_path}
deepspeed --num_gpus 1 triton-bert-benchmark.py --model ${model} --dtype fp16 --trials ${trials} --deepspeed --graphs &> ${log_path}/hf-graph.log
deepspeed --num_gpus 1 triton-bert-benchmark.py --model ${model} --dtype fp16 --trials ${trials} --deepspeed --kernel-inject --graphs &> ${log_path}/ds-graph.log
deepspeed --num_gpus 1 triton-bert-benchmark.py --model ${model} --dtype fp16 --trials ${trials} --deepspeed --kernel-inject --graphs --triton  &> ${log_path}/triton-graph.log


model=roberta-base
log_path=results/${date}_${model}
mkdir -p ${log_path}
deepspeed --num_gpus 1 triton-bert-benchmark.py --model ${model} --dtype fp16 --trials ${trials} --deepspeed --graphs &> ${log_path}/hf-graph.log
deepspeed --num_gpus 1 triton-bert-benchmark.py --model ${model} --dtype fp16 --trials ${trials} --deepspeed --kernel-inject --graphs &> ${log_path}/ds-graph.log
deepspeed --num_gpus 1 triton-bert-benchmark.py --model ${model} --dtype fp16 --trials ${trials} --deepspeed --kernel-inject --graphs --triton  &> ${log_path}/triton-graph.log


model=roberta-large
log_path=results/${date}_${model}
mkdir -p ${log_path}
deepspeed --num_gpus 1 triton-bert-benchmark.py --model ${model} --dtype fp16 --trials ${trials} --deepspeed --graphs &> ${log_path}/hf-graph.log
deepspeed --num_gpus 1 triton-bert-benchmark.py --model ${model} --dtype fp16 --trials ${trials} --deepspeed --kernel-inject --graphs &> ${log_path}/ds-graph.log
deepspeed --num_gpus 1 triton-bert-benchmark.py --model ${model} --dtype fp16 --trials ${trials} --deepspeed --kernel-inject --graphs --triton  &> ${log_path}/triton-graph.log

