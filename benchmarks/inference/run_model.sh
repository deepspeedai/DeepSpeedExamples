set -x

model=$1
dtype=$2
graphs=$3
kernel=$4
gpus=$5

version=0
log_path=results/${model}_${dtype}_${graphs}_${kernel}_${gpus}gpus_v${version}
mkdir -p ${log_path}

params="--dtype $dtype "
if [[ "$graphs" == "true" ]]; then
    params+="--graphs "
fi
if [[ "$kernel" == "true" ]]; then
    params+="--kernel "
fi

echo "baseline $log_path"
deepspeed --num_gpus 1 gpt-bench.py -m "${model}" $params &> ${log_path}/baseline.log

echo "deepspeed $log_path"
deepspeed --num_gpus $gpus gpt-bench.py --deepspeed -m "${model}" $params &> ${log_path}/deepspeed.log