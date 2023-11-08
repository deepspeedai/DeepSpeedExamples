#!/bin/bash

DEPLOYMENT_NAME=${DEPLOYMENT_NAME:-llama2-7b}
VLLM=${VLLM:-""}

CLIENT_NUMS=${CLIENT_NUMS:-1 2 4 6 8 12 16 20 24 28 32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-60}
PROMPT_LENGTH=${PROMPT_LENGTH:-3072}
REQUEST_NUM=${REQUEST_NUM:-512}

LOG_DIR=logs.${DEPLOYMENT_NAME}
mkdir -p ${LOG_DIR}

for client_num in ${CLIENT_NUMS[@]}; do
    RESULT_FILE=${DEPLOYMENT_NAME}_c${client_num}_p${PROMPT_LENGTH}_g${MAX_NEW_TOKENS}.json

    python run_benchmark_client.py -w 1 \
        -d ${DEPLOYMENT_NAME} -n ${REQUEST_NUM} -c ${client_num} \
        -k ${MAX_NEW_TOKENS} -l ${PROMPT_LENGTH} \
        -o ${LOG_DIR}/${RESULT_FILE} \
        ${VLLM} --stream \
        2>&1 | tee ${LOG_DIR}/bench_client_num_c${client_num}_p${PROMPT_LENGTH}_g${MAX_NEW_TOKENS}.log 
done
