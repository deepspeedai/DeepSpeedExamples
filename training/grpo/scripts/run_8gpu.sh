#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# Trains with configs/qwen3_8b_8gpu.json on 8 GPUs. Run scripts/prepare_data.sh first.
set -euo pipefail
cd "$(dirname "$0")/.."

deepspeed --num_gpus 8 main.py --config configs/qwen3_8b_8gpu.json
