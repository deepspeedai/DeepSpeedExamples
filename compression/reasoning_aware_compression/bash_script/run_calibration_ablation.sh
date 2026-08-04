#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# Reproduces the core comparison of the RAC paper: the same model, the same
# pruning algorithm and the same 1M-token budget, pruned three times with
# three different calibration sets.
#
#   c4     generic web text (the SparseGPT default)
#   prompt the task prompts only
#   rac    those prompts plus the dense model's own chain-of-thought
#
# Requires traces from collect_traces.py; run bash_script/run_rac_qwen1_5b.sh
# first (or set TRACES to an existing file).
#
#   cd compression/reasoning_aware_compression
#   TRACES=outputs/traces/DeepSeek-R1-Distill-Qwen-1.5B_math.jsonl \
#       bash bash_script/run_calibration_ablation.sh
set -euo pipefail

MODEL=${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
TRACES=${TRACES:?set TRACES to the JSONL written by collect_traces.py}
SPARSITY=${SPARSITY:-0.5}
NSAMPLES=${NSAMPLES:-512}
SEQLEN=${SEQLEN:-2048}
OUTPUT_ROOT=${OUTPUT_ROOT:-./outputs}

TAG=$(basename "${MODEL}")

for CALIBRATION in c4 prompt rac; do
    OUT=${OUTPUT_ROOT}/checkpoints/${TAG}_${CALIBRATION}_sparsity${SPARSITY}
    echo "=== calibration: ${CALIBRATION} -> ${OUT}"

    EXTRA=()
    if [[ "${CALIBRATION}" == "c4" ]]; then
        # Streams allenai/c4; no traces involved.
        EXTRA=(--dataset allenai/c4 --dataset-config en)
    else
        # Both prompt-only and RAC read the same trace file, so the prompts are
        # identical and the only difference is the CoT tokens.
        EXTRA=(--traces "${TRACES}")
    fi

    python prune.py \
        --model "${MODEL}" \
        --calibration "${CALIBRATION}" \
        --pruning-method sparsegpt \
        --sparsity "${SPARSITY}" \
        --nsamples "${NSAMPLES}" \
        --seqlen "${SEQLEN}" \
        --output "${OUT}" \
        "${EXTRA[@]}"

    echo "MODEL_DIR=${OUT} bash bash_script/eval_math500.sh"
done
