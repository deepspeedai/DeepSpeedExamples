#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# Trains with configs/smoke_qwen3_0.6b.json on 1 GPU. Run scripts/prepare_data.sh first.
set -euo pipefail
cd "$(dirname "$0")/.."

deepspeed --num_gpus 1 main.py --config configs/smoke_qwen3_0.6b.json
