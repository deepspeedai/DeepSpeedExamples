#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# nccl_check.sh: nccl_check.py with torchrun on this node (8 GPUs), in the rdsp venv with
# efa_env.sh; run on both nodes by both.sh (NNODES, NODE_RANK, MASTER_ADDR). NCCL_DEBUG=INFO
# records which network NCCL picked ("NET/OFI ... efa" or "NET/Socket").
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
unset LD_LIBRARY_PATH
. ~/venv/bin/activate
. "$HERE/efa_env.sh"
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET torchrun --nnodes "$NNODES" --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" --master_port 29414 --nproc_per_node 8 "$HERE/nccl_check.py"
