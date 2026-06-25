#!/bin/bash

# Correctness test suite for autosp vs baseline compiled DS-Ulysses.
#
# For each (sp_size, dp_size) x zero_stage configuration:
#   1. Runs baseline (--compile compile) for N steps
#   2. Runs autosp  (--compile autosp)  for N steps
#   3. Compares per-rank losses with validator.py
#
# Usage:
#   ./correctness.sh                    # Default configs
#   ./correctness.sh 2,1 2,2 4,1       # Custom sp,dp pairs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
STEPS=5

# Parse sp,dp pairs from positional args (e.g. 2,1 2,2 4,1)
declare -a CONFIGS=()

if [ $# -gt 0 ]; then
    for arg in "$@"; do
        if [[ "$arg" =~ ^([0-9]+),([0-9]+)$ ]]; then
            CONFIGS+=("$arg")
        else
            echo "Error: invalid config '${arg}'. Expected format: sp,dp (e.g. 2,1)"
            exit 1
        fi
    done
else
    CONFIGS=("2,1" "4,1" "8,1")
fi

ZERO_STAGES=(0 1)

PASS_COUNT=0
FAIL_COUNT=0
TOTAL_COUNT=0
declare -a RESULTS=()

echo ""
echo "================================================================"
echo "  AutoSP Correctness Test Suite"
echo "================================================================"
echo "  Configs (sp,dp): ${CONFIGS[*]}"
echo "  Zero stages:     ${ZERO_STAGES[*]}"
echo "  Steps:           ${STEPS}"
echo "  Output dir:      ${OUTPUT_DIR}"
echo "================================================================"
echo ""

for config in "${CONFIGS[@]}"; do
    sp_size="${config%%,*}"
    dp_size="${config##*,}"

    for zero_stage in "${ZERO_STAGES[@]}"; do
        TEST_NAME="sp${sp_size}_dp${dp_size}_zero${zero_stage}"
        TEST_DIR="${OUTPUT_DIR}/${TEST_NAME}"
        mkdir -p "${TEST_DIR}"

        ((TOTAL_COUNT++))

        echo "----------------------------------------------------------------"
        echo "  Test: sp_size=${sp_size}, dp_size=${dp_size}, zero_stage=${zero_stage}"
        echo "----------------------------------------------------------------"

        # --- Baseline (compiled DS-Ulysses) ---
        echo "  [1/3] Running baseline (--compile compile) ..."
        if ! python3 "${SCRIPT_DIR}/correctness_run.py" \
            --compile compile \
            --sp-size "${sp_size}" \
            --dp-size "${dp_size}" \
            --zero-stage "${zero_stage}" \
            --steps "${STEPS}" \
            --output-file "${TEST_DIR}/baseline.json"; then

            echo "  FAIL: Baseline training failed"
            RESULTS+=("  ${TEST_NAME}: FAIL (baseline training error)")
            ((FAIL_COUNT++))
            echo ""
            continue
        fi

        # --- AutoSP ---
        echo "  [2/3] Running autosp  (--compile autosp)  ..."
        if ! python3 "${SCRIPT_DIR}/correctness_run.py" \
            --compile autosp \
            --sp-size "${sp_size}" \
            --dp-size "${dp_size}" \
            --zero-stage "${zero_stage}" \
            --steps "${STEPS}" \
            --output-file "${TEST_DIR}/autosp.json"; then

            echo "  FAIL: AutoSP training failed"
            RESULTS+=("  ${TEST_NAME}: FAIL (autosp training error)")
            ((FAIL_COUNT++))
            echo ""
            continue
        fi

        # --- Validate ---
        echo "  [3/3] Validating per-rank losses ..."
        if python3 "${SCRIPT_DIR}/validator.py" \
            --baseline "${TEST_DIR}/baseline.json" \
            --autosp "${TEST_DIR}/autosp.json"; then

            RESULTS+=("  ${TEST_NAME}: PASS")
            ((PASS_COUNT++))
        else
            RESULTS+=("  ${TEST_NAME}: FAIL")
            ((FAIL_COUNT++))
        fi

        echo ""
    done
done

# ---- Summary ----
echo "================================================================"
echo "  SUMMARY"
echo "================================================================"
for result in "${RESULTS[@]}"; do
    echo "${result}"
done
echo ""
echo "  Passed: ${PASS_COUNT}/${TOTAL_COUNT}    Failed: ${FAIL_COUNT}/${TOTAL_COUNT}"
echo "================================================================"

if [ "${FAIL_COUNT}" -gt 0 ]; then
    exit 1
fi
exit 0
