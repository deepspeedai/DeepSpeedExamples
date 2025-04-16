PROFILE_DIR=${PROFILE_DIR:-"profile_offload"}
mkdir -p ${PROFILE_DIR}
PROFILE_OPTS="--profile --profile-dir ${PROFILE_DIR}"
COMPILE_OPTS="--compile"
DC_OPTS="--compile --deepcompile"
ACC_OPTS="--gradient-accumulation-steps 1"
AC_OPTS="--activation-checkpointing"

mkdir -p logs

export LOG_BASE="logs_offload"
mkdir -p ${LOG_BASE}

MODEL="meta-llama/Meta-Llama-3-70B-Instruct"
BATCH_SIZE_OPTS=(1)
SEQ_LENGTH_OPTS=(1024)
for BATCH_SIZE in ${BATCH_SIZE_OPTS[@]}; do
    for SEQ_LENGTH in ${SEQ_LENGTH_OPTS[@]}; do
        ARGS="--model ${MODEL} --batch-size ${BATCH_SIZE} --seq-length ${SEQ_LENGTH} ${ACC_OPTS} ${AC_OPTS} ${PROFILE_OPTS}"
        bash ./run.sh --backend deepspeed ${ARGS} --zero-stage 3
        bash ./run.sh --backend deepspeed ${ARGS} --zero-stage 3 --ds-offload
        bash ./run.sh --backend deepspeed ${ARGS} ${DC_OPTS} --zero-stage 3 --eager --passes offload_adam_states
        bash ./run.sh --backend deepspeed ${ARGS} ${DC_OPTS} --zero-stage 3 --eager --passes offload_adam_states_sync
    done
done
