NGPUS=4
HIDDEN_SIZE=32768
NUM_LAYERS=4

TRIALS=10

PIN_MEMORY_OPTS=(0 1)
NON_BLOCKING_OPTS=(0 1)

for i in $(seq 1 $TRIALS); do
    for PIN_MEMORY in "${PIN_MEMORY_OPTS[@]}"; do
        PIN_MEMORY_ARG=""
        if [ $PIN_MEMORY -eq 1 ]; then
            PIN_MEMORY_ARG="--pin_memory"
        fi

        for NON_BLOCKING in "${NON_BLOCKING_OPTS[@]}"; do
            NON_BLOCKING_ARG=""
            if [ $NON_BLOCKING -eq 1 ]; then
                NON_BLOCKING_ARG="--non_blocking"
            fi

            echo "Running iteration $i"
            deepspeed --num_gpus=$NGPUS offload_states.py --hidden_dim $HIDDEN_SIZE --nlayers $NUM_LAYERS $PIN_MEMORY_ARG $NON_BLOCKING_ARG
        done
    done
done
python output_table.py
