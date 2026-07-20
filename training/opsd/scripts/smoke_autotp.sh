#!/bin/bash
# Smoke test: OPSD with AutoTP=2 (requires 2 GPUs)
# Usage: bash scripts/smoke_autotp.sh
export PYTHONPATH=.
deepspeed --num_gpus 2 main.py --config configs/smoke_hybrid_autotp.json
