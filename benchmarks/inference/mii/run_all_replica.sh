RAGGED_BATCH_SIZE=768
PARAM_SIZES=(7b)
REPLICA_NUMS=(1)

declare -A TP_SIZES
TP_SIZES["7b"]="4"
TP_SIZES["13b"]="1"
TP_SIZES["70b"]="4"

for PARAM_SIZE in ${PARAM_SIZES[@]}; do
    IFS=':' read -ra TP_VALUES <<< ${TP_SIZES[${PARAM_SIZE}]}
    for TP in ${TP_VALUES[@]}; do
        for REPL in ${REPLICA_NUMS[@]}; do
            DEPLOYMENT_NAME=llama2-${PARAM_SIZE}-tp${TP}-b${RAGGED_BATCH_SIZE}_repl${REPL}
            python server.py --model_name meta-llama/Llama-2-${PARAM_SIZE}-hf -d ${DEPLOYMENT_NAME} -m ${TP} -r ${REPL} -b ${RAGGED_BATCH_SIZE} start

            REQUEST_NUM=$((256 * ${REPL}))
            DEPLOYMENT_NAME=${DEPLOYMENT_NAME} PROMPT_LENGTH=2600 MAX_NEW_TOKENS=60 CLIENT_NUMS=$((16 * ${REPL})) REQUEST_NUM=$((256 * ${REPL})) bash ./run_bench_client_num.sh

            echo "Stopping server"
            python server.py -d ${DEPLOYMENT_NAME} stop
            sleep 120
        done
    done
done
