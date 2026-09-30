#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

set -euo pipefail

# Run a 50-step checkpoint/resume comparison for Moonlight AutoEP.
#
# The script compares:
#   1. uninterrupted training to step 100;
#   2. native DeepSpeed checkpoint resume at step 50;
#   3. native checkpoint -> universal conversion -> universal resume at step 50.
#
# Override MODEL_NAME, DATASET_NAME, NUM_GPUS, and OUTPUT_ROOT as needed.

NUM_GPUS="${NUM_GPUS:-8}"
AUTOEP_SIZE="${AUTOEP_SIZE:-${NUM_GPUS}}"
ZERO_STAGE="${ZERO_STAGE:-2}"
GPU_INCLUDE="${GPU_INCLUDE:-}"
MODEL_NAME="${MODEL_NAME:-moonshotai/Moonlight-16B-A3B}"
DATASET_NAME="${DATASET_NAME:-tatsu-lab/alpaca}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PWD}/autoep_affine_ir_experiment}"
HF_HOME="${HF_HOME:-}"
DEEPSPEED_REPO="${DEEPSPEED_REPO:-}"
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${EXAMPLE_DIR}/configs/z2_moonlight_autoep_adam.json"
TRAIN="${EXAMPLE_DIR}/finetune_llama.py"
CONVERTER="${DEEPSPEED_REPO}/deepspeed/checkpoint/ds_to_universal.py"
LAUNCHER="${DEEPSPEED_LAUNCHER:-ds}"
BASE_CONFIG="${OUTPUT_ROOT}/autoep_config.json"
UNIVERSAL_CONFIG="${OUTPUT_ROOT}/autoep_universal_config.json"
COMMON_ARGS=(
    --model_name "${MODEL_NAME}"
    --dataset_name "${DATASET_NAME}"
    --batch_size 128
    --max_length 64
    --num_train_epochs 20
    --seed 42
    --skip_weight_export
)

export HF_HOME
export AUTOEP_SIZE
export ZERO_STAGE
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTHONPATH="${DEEPSPEED_REPO}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_${USER:-copilot}}"
mkdir -p "${TRITON_CACHE_DIR}"

mkdir -p "${OUTPUT_ROOT}"
cp "${CONFIG}" "${BASE_CONFIG}"

python - "${BASE_CONFIG}" "${UNIVERSAL_CONFIG}" <<'PY'
import json
import os
import sys

source, target = sys.argv[1:3]
with open(source) as handle:
    config = json.load(handle)
config["train_batch_size"] = 128
config["gradient_accumulation_steps"] = 16
config["zero_optimization"]["stage"] = int(os.environ["ZERO_STAGE"])
config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": False}
if config["zero_optimization"]["stage"] == 3:
    config["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": False}
config["expert_parallel"]["autoep_size"] = int(os.environ["AUTOEP_SIZE"])
with open(source, "w") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
config["checkpoint"] = {"load_universal": True}
with open(target, "w") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
PY

run_train() {
    local output_dir="$1"
    local config="$2"
    shift 2
    local launcher_args=(--num_gpus="${NUM_GPUS}")
    if [[ -n "${GPU_INCLUDE}" ]]; then
        launcher_args=(--include="${GPU_INCLUDE}")
    fi
    "${LAUNCHER}" "${launcher_args[@]}" "${TRAIN}" \
        "${COMMON_ARGS[@]}" \
        --output_dir "${output_dir}" \
        --deepspeed_config "${config}" \
        --checkpoint_steps 50 \
        "$@"
}

echo "Running uninterrupted 100-step baseline..."
run_train "${OUTPUT_ROOT}/baseline_100" "${BASE_CONFIG}" --max_steps 100 \
    --checkpoint_steps 0 \
    > "${OUTPUT_ROOT}/baseline_100.log" 2>&1

echo "Running native checkpoint to step 50..."
run_train "${OUTPUT_ROOT}/native_50" "${BASE_CONFIG}" --max_steps 50 \
    > "${OUTPUT_ROOT}/native_50.log" 2>&1

echo "Resuming from native checkpoint to step 100..."
run_train "${OUTPUT_ROOT}/native_resume_100" "${BASE_CONFIG}" --max_steps 100 \
    --resume_dir "${OUTPUT_ROOT}/native_50" --resume_tag step_50 \
    --checkpoint_steps 0 \
    > "${OUTPUT_ROOT}/native_resume_100.log" 2>&1

echo "Converting step 50 checkpoint to universal format..."
python "${CONVERTER}" \
    --input_folder "${OUTPUT_ROOT}/native_50/step_50" \
    --output_folder "${OUTPUT_ROOT}/native_50/step_50_universal" \
    --num_extract_workers 1 \
    --num_merge_workers 1 \
    --no_strict \
    > "${OUTPUT_ROOT}/convert_universal.log" 2>&1
expert_state_files=("${OUTPUT_ROOT}/native_50/step_50"/layer_*_expert_*_model_states.pt)
if [[ -e "${expert_state_files[0]}" ]]; then
    cp "${expert_state_files[@]}" "${OUTPUT_ROOT}/native_50/step_50_universal/"
fi

echo "Resuming from universal checkpoint to step 100..."
run_train "${OUTPUT_ROOT}/universal_resume_100" "${UNIVERSAL_CONFIG}" --max_steps 100 \
    --resume_dir "${OUTPUT_ROOT}/native_50" --resume_tag step_50_universal \
    --checkpoint_steps 0 \
    > "${OUTPUT_ROOT}/universal_resume_100.log" 2>&1

python - "${OUTPUT_ROOT}" <<'PY'
import csv
import math
import os
import re
import sys

root = sys.argv[1]
pattern = re.compile(r"Step (\d+), Loss: ([0-9eE.+-]+)")
logs = {
    "baseline": os.path.join(root, "baseline_100.log"),
    "native_resume": os.path.join(root, "native_resume_100.log"),
    "universal_resume": os.path.join(root, "universal_resume_100.log"),
}

losses = {}
for name, path in logs.items():
    values = {}
    with open(path) as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                values[int(match.group(1))] = float(match.group(2))
    losses[name] = values

steps = sorted(set.intersection(*(set(values) for values in losses.values())))
steps = [step for step in steps if 51 <= step <= 100]
if not steps:
    raise RuntimeError("No common loss entries were found for steps 51-100.")

csv_path = os.path.join(root, "loss_comparison.csv")
with open(csv_path, "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["step", *losses])
    for step in steps:
        writer.writerow([step, *(losses[name][step] for name in losses)])

summary_path = os.path.join(root, "loss_comparison.txt")
with open(summary_path, "w") as handle:
    for name in ("native_resume", "universal_resume"):
        errors = [
            abs(losses["baseline"][step] - losses[name][step])
            for step in steps
        ]
        max_error = max(errors)
        mean_error = sum(errors) / len(errors)
        handle.write(
            f"{name}: steps={len(steps)} max_abs_error={max_error:.9g} "
            f"mean_abs_error={mean_error:.9g}\n"
        )
    handle.write(f"Compared steps: {steps[0]}-{steps[-1]}\n")

print(f"Loss comparison written to {csv_path}")
print(open(summary_path).read(), end="")
PY

echo "Experiment completed."
echo "Logs and comparison: ${OUTPUT_ROOT}/*.log, ${OUTPUT_ROOT}/loss_comparison.{csv,txt}"
