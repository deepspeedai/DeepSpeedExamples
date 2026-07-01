#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
#
# Launch OPSD training with vLLM rollout.
#
# The vLLM server is started **lazily** as a subprocess by training rank 0
# on first use, so no separate vLLM launch step is required.  The GPUs
# assigned to the vLLM server are controlled by the ROLLOUT_VISIBLE_DEVICE
# environment variable (comma-separated CUDA device indices).  The training
# ranks must run on a *different* set of GPUs so the two don't contend for
# memory.
#
# Default topology: ranks 0..5 train on GPUs 0-5 (ZeRO-3), devices 6-7
# run vLLM with TP=2.  Override via:
#   ROLLOUT_VISIBLE_DEVICE=... NUM_TRAIN_GPUS=.. INCLUDE_GPUS=.. bash ...
set -euo pipefail

CONFIG="${1:-configs/opsd_vllm_disjoint.json}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-6}"
INCLUDE_GPUS="${INCLUDE_GPUS:-0,1,2,3,4,5}"
export ROLLOUT_VISIBLE_DEVICE="${ROLLOUT_VISIBLE_DEVICE:-6,7}"

deepspeed --num_gpus "${NUM_TRAIN_GPUS}" --include "localhost:${INCLUDE_GPUS}" \
    main.py --config "${CONFIG}"
