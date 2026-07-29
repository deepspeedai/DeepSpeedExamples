#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# MATH-500 acc@1 for a pruned (or dense) reasoning checkpoint, following the
# open-r1 evaluation pipeline used in the RAC paper. Reports the total runtime
# too: heavily pruned reasoning models ramble, and the paper's headline is that
# RAC keeps both accuracy *and* decode length close to dense.
#
#   pip install "lighteval[math,vllm]"
#   cd compression/reasoning_aware_compression
#   MODEL_DIR=outputs/checkpoints/... bash bash_script/eval_math500.sh
set -euo pipefail

MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the checkpoint to evaluate}
NUM_GPUS=${NUM_GPUS:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32768}
TASK=${TASK:-"lighteval|math_500|0|0"}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/eval/$(basename "${MODEL_DIR}")}

export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL_ARGS="model_name=${MODEL_DIR},\
dtype=bfloat16,\
trust_remote_code=true,\
max_model_length=${MAX_NEW_TOKENS},\
gpu_memory_utilization=0.8,\
data_parallel_size=${NUM_GPUS},\
generation_parameters={max_new_tokens:${MAX_NEW_TOKENS},temperature:0.6,top_p:0.95}"

START=$SECONDS
lighteval vllm "${MODEL_ARGS}" "${TASK}" \
    --use-chat-template \
    --output-dir "${OUTPUT_DIR}" \
    --save-details
echo "[eval] ${TASK} on ${MODEL_DIR} took $(( (SECONDS - START) / 60 )) min"

# Coding: "extended|lcb:codegeneration|0|0" (needs lighteval[extended-tasks]).
# AIME-25: "lighteval|aime25|0|0".
