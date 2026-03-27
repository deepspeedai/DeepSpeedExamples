#!/bin/bash

# Default parameters
MODEL="meta-llama/Llama-2-7b-chat-hf"
COMPILE="eager"
BACKEND="inductor"
SP_SIZE=2
DP_SIZE=1
BATCH_SIZE=1
SEQ_LENGTH=64
EXTRA_OPTS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --host-ip)
            HOST_IP="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --compile)
            COMPILE="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --seq-length)
            SEQ_LENGTH="$2"
            shift 2
            ;;
        --sp-size)
            SP_SIZE="$2"
            shift 2
            ;;
        --dp-size)
            DP_SIZE="$2"
            shift 2
            ;;
        --num-layers)
            EXTRA_OPTS="${EXTRA_OPTS} --num_layers $2"
            shift 2
            ;;
        *)
            EXTRA_OPTS="${EXTRA_OPTS} $1"
            shift
            ;;
    esac
done

if [[ "$COMPILE" != "eager" && "$COMPILE" != "compile" && "$COMPILE" != "autosp" && "$COMPILE" != "ringattn" ]]; then
    echo "Invalid compile mode: $COMPILE. Choose from eager, compile, autosp, ringattn."
    exit 1
fi

if [[ -z "${HOST_IP}" ]]; then
    HOST_IP=$(hostname -i | awk '{print $1}')
fi

PORT=$(python3 -c "import socket; s = socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

NUM_PROCESSES=$((SP_SIZE * DP_SIZE))

CONFIG_FILE="configs/torchcompile_config.yaml"
if [ "${COMPILE}" == "autosp" ]; then
    CONFIG_FILE="configs/autosp_config.yaml"
fi

mkdir -p logs

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=logs/log_${COMPILE}_sp${SP_SIZE}_dp${DP_SIZE}_seq${SEQ_LENGTH}_${TIMESTAMP}.log

# Print configuration
echo ""
echo "================================================================"
echo "Configuration"
echo "================================================================"
echo "HOST_IP: ${HOST_IP}"
echo "PORT: ${PORT}"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"
echo "MODEL: ${MODEL}"
echo "COMPILE: ${COMPILE}"
echo "BACKEND: ${BACKEND}"
echo "SP_SIZE: ${SP_SIZE}"
echo "DP_SIZE: ${DP_SIZE}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "SEQ_LENGTH: ${SEQ_LENGTH}"
echo "LOG_FILE: ${LOG_FILE}"
echo "================================================================"
echo ""

export NCCL_DEBUG=WARN

# Launch training
accelerate launch \
    --main_process_ip ${HOST_IP} \
    --main_process_port ${PORT} \
    --num_machines 1 \
    --num_processes ${NUM_PROCESSES} \
    --machine_rank 0 \
    --config_file ${CONFIG_FILE} \
    run.py \
    --model_name "${MODEL}" \
    --batch_size ${BATCH_SIZE} \
    --seq_length ${SEQ_LENGTH} \
    --sp_size ${SP_SIZE} \
    --dp_size ${DP_SIZE} \
    --backend ${BACKEND} \
    --compile ${COMPILE} \
    ${EXTRA_OPTS} \
    2>&1 | tee ${LOG_FILE}
