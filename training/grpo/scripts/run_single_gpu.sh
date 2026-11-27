#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# Trains with configs/qwen3_4b_instruct_1gpu.json on 1 GPU. Run scripts/prepare_data.sh first.
set -euo pipefail
cd "$(dirname "$0")/.."

deepspeed --num_gpus 1 main.py --config configs/qwen3_4b_instruct_1gpu.json
