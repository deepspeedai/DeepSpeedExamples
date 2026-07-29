#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# End-to-end RAC on DeepSeek-R1-Distill-Qwen-1.5B: collect on-policy CoT
# traces, prune to 50% with SparseGPT calibrated on them, then evaluate.
# Runs on a single 80GB GPU; trace collection dominates the wall clock.
#
#   cd compression/reasoning_aware_compression
#   bash bash_script/run_rac_qwen1_5b.sh
set -euo pipefail

MODEL=${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
DATASET=${DATASET:-open-r1/OpenR1-Math-220k}
PROMPT_COLUMN=${PROMPT_COLUMN:-problem}
SPARSITY=${SPARSITY:-0.5}
TRACE_TOKENS=${TRACE_TOKENS:-1000000}
NSAMPLES=${NSAMPLES:-512}
SEQLEN=${SEQLEN:-2048}
OUTPUT_ROOT=${OUTPUT_ROOT:-./outputs}

TAG=$(basename "${MODEL}")
TRACES=${OUTPUT_ROOT}/traces/${TAG}_math.jsonl
PRUNED=${OUTPUT_ROOT}/checkpoints/${TAG}_rac_sparsity${SPARSITY}

# ---------------------------------------------------------------- phase I --
# On-policy rollouts from the *dense* model. Skipped if the traces exist, so
# the pruning sweep below can be re-run cheaply.
if [[ -s "${TRACES}" ]]; then
    echo "[run_rac] reusing traces at ${TRACES}"
else
    python collect_traces.py \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --prompt-column "${PROMPT_COLUMN}" \
        --trace-tokens "${TRACE_TOKENS}" \
        --max-new-tokens 8192 \
        --temperature 0.6 \
        --top-p 0.95 \
        --output "${TRACES}"
fi

# --------------------------------------------------------------- phase II --
python prune.py \
    --model "${MODEL}" \
    --calibration rac \
    --traces "${TRACES}" \
    --pruning-method sparsegpt \
    --sparsity "${SPARSITY}" \
    --nsamples "${NSAMPLES}" \
    --seqlen "${SEQLEN}" \
    --output "${PRUNED}" \
    --save-report

echo "[run_rac] pruned checkpoint: ${PRUNED}"
echo "[run_rac] evaluate with: MODEL_DIR=${PRUNED} bash bash_script/eval_math500.sh"
