#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# node_prep.sh: one node ready for the 35B runs (prep.sh runs it on both nodes at once): the rdsp
# environment and NeMo container (bench/setup_node.sh), TorchTitan's environment
# (torchtitan/scripts/setup_box.sh) and the Qwen3.5-35B-A3B weights (~72 GB) side by side, then a
# record of the node's GPUs and EFA devices in ~/hardware.txt. Logs in ~/.
set -uo pipefail   # no -e: each step checks its own exit code and reports which one failed
RDSP=$(cd "$(dirname "$0")/../../.." && pwd)
bash "$RDSP/bench/setup_node.sh" > ~/setup.log 2>&1 || { echo "SETUP FAILED"; exit 1; }
bash "$RDSP/bench/torchtitan/scripts/setup_box.sh" > ~/tt_setup.log 2>&1 &
TT=$!
(unset LD_LIBRARY_PATH; . ~/venv/bin/activate; HF_HOME=~/hf python -c "import huggingface_hub as h; \
 h.snapshot_download('Qwen/Qwen3.5-35B-A3B', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])") \
  > ~/download.log 2>&1 || { echo "DOWNLOAD FAILED"; exit 1; }
wait $TT || { echo "TORCHTITAN SETUP FAILED"; exit 1; }
{ hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; nvidia-smi topo -m
  echo "EFA devices: $(ls /sys/class/infiniband 2>/dev/null | wc -l)"
  /opt/amazon/efa/bin/fi_info -p efa 2>/dev/null | grep -m3 -E "provider|fabric|domain"
  ls /opt/amazon/ofi-nccl/lib 2>/dev/null | head -3; lscpu | head -20; } > ~/hardware.txt 2>&1
touch ~/.rdsp_node_prep_done   # prep.sh skips this node next time
echo "NODE PREP DONE"
