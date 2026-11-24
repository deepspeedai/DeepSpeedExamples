#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

set -euo pipefail

# HF checkpoint -> AutoEP ZeRO training -> rollout at a new expert layout, once through a
# universal checkpoint (UCP) and once through AffineIR transfers, with bitwise oracles.
#
# MODEL_NAME must be a local HF checkpoint directory. Override NUM_GPUS, NUM_LAYERS,
# TRAIN_AUTOEP_SIZE, ROLLOUT_AUTOEP_SIZE, ZERO_STAGES, and OUTPUT_ROOT as needed.
# The UCP path writes a full training checkpoint and its universal form under OUTPUT_ROOT.

NUM_GPUS="${NUM_GPUS:-8}"
NUM_LAYERS="${NUM_LAYERS:-4}"
TRAIN_AUTOEP_SIZE="${TRAIN_AUTOEP_SIZE:-4}"
ROLLOUT_AUTOEP_SIZE="${ROLLOUT_AUTOEP_SIZE:-8}"
ZERO_STAGES="${ZERO_STAGES:-2 3}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME to a local HF checkpoint directory}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PWD}/autoep_rollout_e2e}"
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p "${OUTPUT_ROOT}"

for stage in ${ZERO_STAGES}; do
    # A universal checkpoint converted from ZeRO-3 keeps ZeRO-3 model file names, which the
    # ZeRO-1 rollout engine cannot load, so ZeRO-3 runs only the AffineIR path.
    paths="ucp,affine"
    if [ "${stage}" = "3" ]; then
        paths="affine"
    fi
    torchrun --nproc_per_node "${NUM_GPUS}" "${EXAMPLE_DIR}/autoep_rollout_e2e.py" \
        --model_name "${MODEL_NAME}" \
        --num_layers "${NUM_LAYERS}" \
        --train_autoep_size "${TRAIN_AUTOEP_SIZE}" \
        --rollout_autoep_size "${ROLLOUT_AUTOEP_SIZE}" \
        --train_zero_stage "${stage}" \
        --paths "${paths}" \
        --work_dir "${OUTPUT_ROOT}/zero${stage}_checkpoint" \
        --output "${OUTPUT_ROOT}/zero${stage}.json" 2>&1 | tee "${OUTPUT_ROOT}/zero${stage}.log"
    rm -rf "${OUTPUT_ROOT}/zero${stage}_checkpoint"
done
