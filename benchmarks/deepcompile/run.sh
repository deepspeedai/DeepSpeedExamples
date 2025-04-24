#!/bin/bash

NUM_NODES=${NUM_NODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
NUM_PROCESSES=$((${NUM_NODES} * ${NGPUS_PER_NODE}))

BACKEND="deepspeed"
MODEL="meta-llama/Meta-Llama-3-8B"
ZERO_STAGE=3
COMPILE=0
PASSES="ALL"
EXTRA_OPTS=""

EAGER=0
DEEPCOMPILE=0
GRADIENT_ACCUMULATION_STEPS=1
ACTIVATION_CHECKPOINTING=1
BATCH_SIZE=1
SEQ_LENGTH=512
DEBUG_LOG=0
SYNC_BEFORE_REDUCE=0
SYNC_AFTER_REDUCE=0
SYNC_BEFORE_ALLGATHER=0
SYNC_AFTER_ALLGATHER=0

echo "NUM_NODES: ${NUM_NODES} NGPUS_PER_NODE: ${NGPUS_PER_NODE} NUM_PROCESSES: ${NUM_PROCESSES}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --host-ip)
            HOST_IP="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --zero-stage)
            ZERO_STAGE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            EXTRA_OPTS="${EXTRA_OPTS} --batch_size $2"
            shift 2
            ;;
        --seq-length)
            SEQ_LENGTH="$2"
            EXTRA_OPTS="${EXTRA_OPTS} --seq_length $2"
            shift 2
            ;;
        --gradient-accumulation-steps)
            GRADIENT_ACCUMULATION_STEPS="$2"
            EXTRA_OPTS="${EXTRA_OPTS} --gradient_accumulation_steps $2"
            shift 2
            ;;
        --activation-checkpointing)
            ACTIVATION_CHECKPOINTING=1
            EXTRA_OPTS="${EXTRA_OPTS} --activation_checkpointing"
            shift
            ;;   
        --compile)
            COMPILE=1
            EXTRA_OPTS="${EXTRA_OPTS} $1"
            shift
            ;;
        --eager)
            EAGER=1
            EXTRA_OPTS="${EXTRA_OPTS} --backend eager"
            shift
            ;;
        --deepcompile)
            DEEPCOMPILE=1
            shift
            ;;
        --passes)
            PASSES="$2"
            EXTRA_OPTS="${EXTRA_OPTS} $1 $2"
            shift 2
            ;;
        --profile)
            EXTRA_OPTS="${EXTRA_OPTS} $1"
            shift
            ;;
        --profile-dir)
            EXTRA_OPTS="${EXTRA_OPTS} --profile_dir $2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --num-layers)
            EXTRA_OPTS="${EXTRA_OPTS} --num_layers $2"
            shift 2
            ;;
        --attn-impl)
            EXTRA_OPTS="${EXTRA_OPTS} --attn_impl $2"
            shift 2
            ;;
        --eval)
            EXTRA_OPTS="${EXTRA_OPTS} --eval"
            shift
            ;;
        --debug-log)
            DEBUG_LOG=1
            shift
            ;;
        --sync-before-reduce)
            SYNC_BEFORE_REDUCE=1
            shift
            ;;
        --sync-after-reduce)
            SYNC_AFTER_REDUCE=1
            shift
            ;;
        --sync-before-allgather)
            SYNC_BEFORE_ALLGATHER=1
            shift
            ;;
        --sync-after-allgather)
            SYNC_AFTER_ALLGATHER=1
            shift
            ;;
        *)
            EXTRA_OPTS="${EXTRA_OPTS} $1"
            shift
            ;;
    esac
done



export NCCL_DEBUG=WARN

CONFIG_TEMPLATE=configs/ds_config.yaml.template
if [ "${BACKEND}" == "fsdp" ]; then
    CONFIG_TEMPLATE=configs/fsdp_config.yaml.template
elif [ "${BACKEND}" == "ddp" ]; then
    CONFIG_TEMPLATE=configs/ddp_config.yaml.template
elif [ "${BACKEND}" == "singlegpu" ]; then
    CONFIG_TEMPLATE=configs/singlegpu_config.yaml.template
elif [ "${BACKEND}" != "deepspeed" ]; then
    echo "Invalid backend: ${BACKEND}"
    exit 1
fi

if [ "${BACKEND}" != "deepspeed" ]; then
    ZERO_STAGE=0
fi

echo "HOST_IP: ${HOST_IP}"
echo "NUM_NODES: ${NUM_NODES}"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"
echo "BACKEND: ${BACKEND}"
echo "ZERO_STAGE: ${ZERO_STAGE}"
echo "MODEL: ${MODEL}"
echo "GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}"
echo "EXTRA_OPTS: ${EXTRA_OPTS}"

MACHINE_RANK=$(hostname | sed 's/[^0-9]*//g')

python generate_conf.py \
    --machine_rank ${MACHINE_RANK} \
    --num_machines ${NUM_NODES} \
    --num_processes ${NUM_PROCESSES} \
    --zero_stage ${ZERO_STAGE} \
    --template_file ${CONFIG_TEMPLATE} \
    --output_file configs/config.yaml

GAS_OPTS="--gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}"

if [ "${BACKEND}" == "deepspeed" ]; then
    DEEPCOMPILE_OPTS=""
    if [ "${DEEPCOMPILE}" == "1" ]; then
        DEEPCOMPILE_OPTS="--deepcompile"
    fi

    DEBUG_LOG_OPTS=""
    if [ "${DEBUG_LOG}" == "1" ]; then
        DEBUG_LOG_OPTS="--debug_log"
    fi

    SYNC_BEFORE_REDUCE_OPTS=""
    if [ "${SYNC_BEFORE_REDUCE}" == "1" ]; then
        SYNC_BEFORE_REDUCE_OPTS="--sync_before_reduce"
    fi
    
    SYNC_AFTER_REDUCE_OPTS=""
    if [ "${SYNC_AFTER_REDUCE}" == "1" ]; then
        SYNC_AFTER_REDUCE_OPTS="--sync_after_reduce"
    fi

    SYNC_BEFORE_ALLGATHER_OPTS=""
    if [ "${SYNC_BEFORE_ALLGATHER}" == "1" ]; then
        SYNC_BEFORE_ALLGATHER_OPTS="--sync_before_allgather"
    fi

    SYNC_AFTER_ALLGATHER_OPTS=""
    if [ "${SYNC_AFTER_ALLGATHER}" == "1" ]; then
        SYNC_AFTER_ALLGATHER_OPTS="--sync_after_allgather"
    fi

    python generate_conf.py \
        --machine_rank ${MACHINE_RANK} \
        --num_machines ${NUM_NODES} \
        --num_processes ${NUM_PROCESSES} \
        --zero_stage ${ZERO_STAGE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        ${DEEPCOMPILE_OPTS} ${DEBUG_LOG_OPTS} \
        ${SYNC_BEFORE_REDUCE_OPTS} ${SYNC_AFTER_REDUCE_OPTS} \
        ${SYNC_BEFORE_ALLGATHER_OPTS} ${SYNC_AFTER_ALLGATHER_OPTS} \
        --template_file configs/ds_config.json.template \
        --output_file configs/ds_config.json
fi

#replace , with _ in PASSES
PASSES=$(echo $PASSES | tr ',' '_')
LOG_DIR=logs
mkdir -p ${LOG_DIR}
LOG_FILE=${LOG_DIR}/debug_n${MACHINE_RANK}_${MODEL##*/}_${BACKEND}_np${NUM_PROCESSES}z${ZERO_STAGE}c${COMPILE}dc${DEEPCOMPILE}E${EAGER}b${BATCH_SIZE}seq${SEQ_LENGTH}g${GRADIENT_ACCUMULATION_STEPS}a${ACTIVATION_CHECKPOINTING}p${PASSES}.log
echo "Logging to ${LOG_FILE}"

${HOME}/.local/bin/accelerate launch --main_process_ip ${HOST_IP} --main_process_port 12345 \
--num_machines ${NUM_NODES} --num_processes ${NUM_PROCESSES} --machine_rank ${MACHINE_RANK} \
--config_file configs/config.yaml \
run_bench_lm.py \
--model_name "${MODEL}" \
--zero_stage ${ZERO_STAGE} \
${GAS_OPTS} \
${EXTRA_OPTS} \
2>&1 | tee ${LOG_FILE}
