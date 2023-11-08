RAGGED_BATCH_SIZE=768
PARAM_SIZES=(7b 13b 70b)

declare -A TP_SIZES
TP_SIZES["7b"]="1"
TP_SIZES["13b"]="1:2:4"
TP_SIZES["70b"]="4:8"

for PARAM_SIZE in ${PARAM_SIZES[@]}; do
    
    IFS=':' read -ra TP_VALUES <<< ${TP_SIZES[${PARAM_SIZE}]}
    for TP in ${TP_VALUES[@]}; do
        DEPLOYMENT_NAME=vllm-llama2-${PARAM_SIZE}-tp${TP}
        python -m vllm.entrypoints.api_server --host 127.0.0.1 --port 26500 --tensor-parallel-size ${TP} --model meta-llama/Llama-2-${PARAM_SIZE}-hf &
        sleep 60

        DEPLOYMENT_NAME=${DEPLOYMENT_NAME} PROMPT_LENGTH=2600 MAX_NEW_TOKENS=60 VLLM="--vllm" bash ./run_benchmark_client.sh
        DEPLOYMENT_NAME=${DEPLOYMENT_NAME} PROMPT_LENGTH=2600 MAX_NEW_TOKENS=128 VLLM="--vllm" bash ./run_benchmark_client.sh
        DEPLOYMENT_NAME=${DEPLOYMENT_NAME} PROMPT_LENGTH=1200 MAX_NEW_TOKENS=60 VLLM="--vllm" bash ./run_benchmark_client.sh
        DEPLOYMENT_NAME=${DEPLOYMENT_NAME} PROMPT_LENGTH=1200 MAX_NEW_TOKENS=128 VLLM="--vllm" bash ./run_benchmark_client.sh

        echo "Stopping server"
        pkill -u ${USER} -f vllm.entrypoints.api_server
        sleep 30
    done
done
