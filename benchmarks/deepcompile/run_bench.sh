PROFILE_DIR=${PROFILE_DIR:-"profiles"}
mkdir -p ${PROFILE_DIR}
PROFILE_OPTS="--profile --profile-dir ${PROFILE_DIR}"
COMPILE_OPTS="--compile"
DC_OPTS="--compile --deepcompile"
ACC_OPTS="--gradient-accumulation-steps 1"
AC_OPTS="--activation-checkpointing"

export NUM_NODES=${NUM_NODES:-4}

MODEL="meta-llama/Meta-Llama-3-70B-Instruct"
BATCH_SIZE_OPTS=(1 2 4)
SEQ_LENGTH_OPTS=(512 1024 2048)
for BATCH_SIZE in ${BATCH_SIZE_OPTS[@]}; do
    for SEQ_LENGTH in ${SEQ_LENGTH_OPTS[@]}; do
        # skip if batch size is 4 and seq length is 2048, as it causes OOM
        if [ ${BATCH_SIZE} -eq 4 ] && [ ${SEQ_LENGTH} -eq 2048 ]; then
            continue
        fi

        ARGS="--model ${MODEL} --batch-size ${BATCH_SIZE} --seq-length ${SEQ_LENGTH} ${ACC_OPTS} ${AC_OPTS}"
        bash ./run_multinode.sh --backend deepspeed ${ARGS}
        bash ./run_multinode.sh --backend deepspeed ${ARGS} ${COMPILE_OPTS}
        bash ./run_multinode.sh --backend fsdp ${ARGS}
        bash ./run_multinode.sh --backend fsdp ${ARGS} ${COMPILE_OPTS}
        bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS} --passes prefetch,selective_gather
        bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS} --passes prefetch
        bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS} --passes selective_gather

        cp -r logs ${PROFILE_DIR}/
    done
done

MODEL="mistralai/Mixtral-8x7B-v0.1"
BATCH_SIZE_OPTS=(1 2 4)
SEQ_LENGTH_OPTS=(512 1024 2048)
for BATCH_SIZE in ${BATCH_SIZE_OPTS[@]}; do
    for SEQ_LENGTH in ${SEQ_LENGTH_OPTS[@]}; do
        # skip if batch size is 4 and seq length is 2048, as it causes OOM
        ARGS="--model ${MODEL} --batch-size ${BATCH_SIZE} --seq-length ${SEQ_LENGTH} ${ACC_OPTS} ${AC_OPTS}"
        bash ./run_multinode.sh --backend deepspeed ${ARGS}
        bash ./run_multinode.sh --backend deepspeed ${ARGS} ${COMPILE_OPTS}
        bash ./run_multinode.sh --backend fsdp ${ARGS}
        bash ./run_multinode.sh --backend fsdp ${ARGS} ${COMPILE_OPTS}
        bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS} --passes prefetch,selective_gather
        bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS} --passes prefetch
        bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS} --passes selective_gather

        cp -r logs ${PROFILE_DIR}/
    done
done

