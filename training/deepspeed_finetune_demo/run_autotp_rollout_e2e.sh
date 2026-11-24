#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

set -euo pipefail

# Train with AutoTP + ZeRO, then sync the weights into rollout engines at another TP size
# through gather and through affine storage maps, for ZeRO-2 and ZeRO-3.
#
# Override MODEL_NAME, NUM_GPUS, TRAIN_TP_SIZE, ROLLOUT_TP_SIZE, ZERO_STAGES, PATHS,
# REPLICA_POLICY, and OUTPUT_ROOT as needed.

NUM_GPUS="${NUM_GPUS:-8}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME to a local Hugging Face checkpoint directory}"
TRAIN_TP_SIZE="${TRAIN_TP_SIZE:-2}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
ZERO_STAGES="${ZERO_STAGES:-2 3}"
PATHS="${PATHS:-ucp,gather,affine}"
REPLICA_POLICY="${REPLICA_POLICY:-balanced}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PWD}/autotp_rollout_e2e}"
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${OUTPUT_ROOT}"

for stage in ${ZERO_STAGES}; do
    tag="z${stage}_tp${TRAIN_TP_SIZE}_to_tp${ROLLOUT_TP_SIZE}"
    torchrun --nproc_per_node "${NUM_GPUS}" "${EXAMPLE_DIR}/autotp_rollout_e2e.py" \
        --model_name "${MODEL_NAME}" \
        --train_tp_size "${TRAIN_TP_SIZE}" \
        --rollout_tp_size "${ROLLOUT_TP_SIZE}" \
        --train_zero_stage "${stage}" \
        --paths "${PATHS}" \
        --replica_policy "${REPLICA_POLICY}" \
        --work_dir "${OUTPUT_ROOT}/${tag}_ckpt" \
        --output "${OUTPUT_ROOT}/${tag}.json" \
        "$@" 2>&1 | tee "${OUTPUT_ROOT}/${tag}.log"
    rm -rf "${OUTPUT_ROOT}/${tag}_ckpt"
done
