PROFILE_DIR=${PROFILE_DIR:-profiles}
mkdir -p ${PROFILE_DIR}
PROFILE_OPTS="--profile --profile-dir ${PROFILE_DIR}"
COMPILE_OPTS="--compile"
DC_OPTS="--compile --deepcompile"
ACC_OPTS="--gradient-accumulation-steps 1"
AC_OPTS="--activation-checkpointing"

export NUM_NODES=${NUM_NODES:-4}

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
BATCH_SIZE_OPTS=(1 2 4)
SEQ_LENGTH_OPTS=(512 1024 2048)
for BATCH_SIZE in ${BATCH_SIZE_OPTS[@]}; do
    for SEQ_LENGTH in ${SEQ_LENGTH_OPTS[@]}; do
        ARGS="--model ${MODEL} --batch-size ${BATCH_SIZE} --seq-length ${SEQ_LENGTH} --zero-stage 1 ${ACC_OPTS} ${AC_OPTS}"
        bash ./run_multinode.sh --backend deepspeed ${ARGS}
        bash ./run_multinode.sh --backend deepspeed ${ARGS} ${COMPILE_OPTS}
        bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS}

        cp -r logs ${PROFILE_DIR}/
    done
done
