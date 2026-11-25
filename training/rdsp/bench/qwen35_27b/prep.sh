#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# prep.sh, on node 1 (docs/GETTING_STARTED.md lists what must be in place): both nodes ready.
#   1. bench/qwen35_35b/prep.sh: environments, NeMo container, 35B weights, Bridge's 35B
#      checkpoint, the 35B rows at 128 a step on both nodes, NCCL check, Ray cluster
#   2. the dense 27B: weights on both nodes, Bridge's checkpoint on node 1 (its one-node runs),
#      Bridge's rows at 64 a step (the one-node runs' rows) on both nodes
# Idempotent like the 35B prep. Logs in logs/prep/.
set -uo pipefail   # no -e: each step checks its own exit code and reports which one failed
RDSP=~/rdsp
. ~/env2.sh
LOGS=$RDSP/bench/qwen35_27b/logs/prep; mkdir -p "$LOGS"
SSH2="ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no -o BatchMode=yes ubuntu@$PRIV2"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
step() { echo "--- $* $(date -u +%H:%M:%S)"; }

step "START 35B prep"
bash "$RDSP/bench/qwen35_35b/prep.sh" > "$LOGS/prep35.log" 2>&1
grep -q "^=== PREP DONE" "$LOGS/prep35.log" || { mark "PREP FAILED 35B (logs/prep/prep35.log)"; exit 1; }
grep -E "bus bandwidth|Selected|network" "$LOGS/prep35.log" | head -5
step "END 35B prep"

step "START 27B weights"
DL="unset LD_LIBRARY_PATH; . ~/venv/bin/activate; HF_HOME=~/hf python -c \"import huggingface_hub as h; \
h.snapshot_download('Qwen/Qwen3.5-27B', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])\""
$SSH2 "$DL" < /dev/null > "$LOGS/download27b-node2.log" 2>&1 &
D2=$!
bash -c "$DL" > "$LOGS/download27b-node1.log" 2>&1; rc1=$?
wait $D2; rc2=$?
[ $rc1 = 0 ] && [ $rc2 = 0 ] || { mark "PREP FAILED 27B download: node 1 exit $rc1, node 2 exit $rc2"; exit 1; }
step "END 27B weights"

step "START 27B convert and export"
MODEL=Qwen3.5-27B
CONTAINER="sudo docker run --gpus all --rm --ipc=host --shm-size=64g --ulimit memlock=-1 \
  -v $HOME/hf:/root/.cache/huggingface -v $HOME/mimo:/workspace -v $RDSP:/rdsp \
  -e HF_HOME=/root/.cache/huggingface -e PYTHONUNBUFFERED=1 -w /opt/Megatron-Bridge nvcr.io/nvidia/nemo:26.08"
if [ ! -f "$HOME/mimo/std/$MODEL.done" ]; then
  $CONTAINER bash -c "python scripts/conversion/setup_conversion.py import --hf-model Qwen/$MODEL \
    --megatron-path /workspace/std/$MODEL" > "$LOGS/convert27b.log" 2>&1; rc=$?
  sudo chown -R "$(id -u):$(id -g)" "$HOME/hf" "$HOME/mimo"
  [ $rc = 0 ] || { mark "PREP FAILED 27B convert exit $rc"; exit 1; }
  touch "$HOME/mimo/std/$MODEL.done"
fi
if [ ! -f "$HOME/mimo/cord_steps-$MODEL/stats.json" ]; then
  env MODEL=$MODEL RECIPE=qwen35_vl_27b_sft_16gpu_h100_bf16_config GBS=64 STEPS=50 LOGS="$LOGS" \
    bash "$RDSP/bench/common/runs.sh" export > /dev/null 2>&1; rc=$?
  mv "$LOGS/export.log" "$LOGS/export27b.log" 2>/dev/null
  [ $rc = 0 ] && [ -f "$HOME/mimo/cord_steps-$MODEL/stats.json" ] || { mark "PREP FAILED 27B export exit $rc"; exit 1; }
fi
cat "$HOME/mimo/cord_steps-$MODEL/stats.json"; echo   # 65,003 real tokens a step on 2026-10-17
$SSH2 "mkdir -p ~/mimo" < /dev/null
rsync -a -e "ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no" "$HOME/mimo/cord_steps-$MODEL/" \
  "ubuntu@$PRIV2:mimo/cord_steps-$MODEL/" || { mark "PREP FAILED copy 27B rows to node 2"; exit 1; }
step "END 27B convert and export"
mark "PREP DONE"
