#!/bin/sh

MODEL_NAME=facebook/opt-6.7b # ONLY OPT AND BLOOM MODELS ARE SUPPORTED FOR NOW
BATCHSIZE=80 # batch size
PROMPT_LEN=512 # the length of the prompt
GEN_LEN=32 # number of tokens to generate

USE_CPU_OFFLOAD=1 # whether to use model weights cpu offloading when running with deepspeed zero inference
USE_KV_OFFLOAD=1 # whether to use kv cache cpu offloading when running with deepspeed zero inference
USE_HF_MODEL=0 # whether to use the original HF model(no kv cache offloading support) or not
USE_QUANT=0 # whether to use model weigths quantization or not

if [ $USE_CPU_OFFLOAD -eq 1 ]; then
    CPU_OFFLOAD="--cpu-offload"
else
    CPU_OFFLOAD=""
fi

if [ $USE_KV_OFFLOAD -eq 1 ]; then
    KV_OFFLOAD="--kv-offload"
else
    KV_OFFLOAD=""
fi

if [ $USE_HF_MODEL -eq 1 ]; then
    HF_MODEL="--hf-model"
else
    HF_MODEL=""
fi

if [ $USE_HF_MODEL -eq 1 ]; then
    QUANT_BTIS="--quant_bits"
else
    QUANT_BTIS=""
fi


# weight/kv cache cpu examples with small models
# deepspeed --num_gpus 1 run_model.py --model bigscience/bloom-560m --batch-size 3 --cpu-offload  --kv-offload
# deepspeed --num_gpus 1 run_model.py --model facebook/opt-125m --batch-size 3 --cpu-offload --kv-offload

deepspeed --num_gpus 1 run_model.py --model ${MODEL_NAME} --batch-size ${BATCHSIZE} --cpu-offload --prompt-len ${PROMPT_LEN} --gen-len ${GEN_LEN} ${CPU_OFFLOAD} ${KV_OFFLOAD} ${QUANT_BTIS}
