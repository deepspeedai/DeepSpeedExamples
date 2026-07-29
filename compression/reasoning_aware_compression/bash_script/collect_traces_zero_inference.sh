#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# Trace collection for models that do not fit in aggregate GPU memory, using
# ZeRO-Inference: ZeRO-3 shards the parameters across ranks and streams them
# from CPU (or NVMe) as each layer runs. All ranks generate the same batch in
# lockstep (synced_gpus) and rank 0 writes the JSONL.
#
# See inference/huggingface/zero_inference/README.md in this repo for the
# memory/throughput characteristics of the offload paths.
#
#   cd compression/reasoning_aware_compression
#   NUM_GPUS=2 MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
#       bash bash_script/collect_traces_zero_inference.sh
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-2}
MODEL=${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}
DATASET=${DATASET:-open-r1/OpenR1-Math-220k}
PROMPT_COLUMN=${PROMPT_COLUMN:-problem}
TRACE_TOKENS=${TRACE_TOKENS:-1000000}
BATCH_SIZE=${BATCH_SIZE:-4}
OUTPUT=${OUTPUT:-./outputs/traces/$(basename "${MODEL}")_math.jsonl}
# Set NVME_OFFLOAD_DIR to spill parameters to NVMe instead of CPU memory.
NVME_OFFLOAD_DIR=${NVME_OFFLOAD_DIR:-}

OFFLOAD_ARGS=(--cpu-offload)
if [[ -n "${NVME_OFFLOAD_DIR}" ]]; then
    OFFLOAD_ARGS=(--nvme-offload-dir "${NVME_OFFLOAD_DIR}")
fi

deepspeed --num_gpus "${NUM_GPUS}" collect_traces.py \
    --backend deepspeed \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --prompt-column "${PROMPT_COLUMN}" \
    --trace-tokens "${TRACE_TOKENS}" \
    --max-new-tokens 8192 \
    --batch-size "${BATCH_SIZE}" \
    --output "${OUTPUT}" \
    "${OFFLOAD_ARGS[@]}"

echo "[traces] written to ${OUTPUT}"
