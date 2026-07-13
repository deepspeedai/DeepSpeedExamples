#!/usr/bin/env bash
# DeepSpeed Team
#
# Assumes you're cd'd into examples/opsd/.
set -euo pipefail

CONFIG="${1:-configs/opsd_hybrid_engine.json}"
NUM_GPUS="${NUM_GPUS:-8}"

deepspeed --num_gpus "${NUM_GPUS}" main.py --config "${CONFIG}"
