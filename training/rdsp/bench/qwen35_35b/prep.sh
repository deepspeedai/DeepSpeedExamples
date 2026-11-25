#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# prep.sh, on node 1 with ~/env2.sh and ~/key2.pem in place (docs/GETTING_STARTED.md): both
# nodes ready for the 35B runs. Logs in logs/prep/.
#   1. node_prep.sh on both nodes at once: environments, NeMo container, 35B weights, hardware
#   2. node 1: Bridge's checkpoint (HF -> Megatron, the standard path; MIMO is not run) and
#      Bridge's rows exported at 128 per step, 50 steps; both copied to node 2
#   3. NCCL all-reduce over both nodes (EFA) and the network NCCL picked
#   4. the Ray cluster over both nodes, for rdsp
# SKIP_CONVERT=1: no Bridge checkpoint (rdsp-only rounds; the rows export needs only the container).
# Steps already done (their outputs in place) are skipped, so a failed prep restarts where it
# stopped: ~/.rdsp_node_prep_done on each node, ~/mimo/std/$MODEL.done, the export's stats.json.
set -uo pipefail   # no -e: each step checks its own exit code and reports which one failed
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../common/two_node.sh"
. ~/env2.sh
LOGS=$HERE/logs/prep; mkdir -p "$LOGS"
SSH2="ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no -o BatchMode=yes ubuntu@$PRIV2"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }  # "=== " lines are the outcome: DONE, FAILED
step() { echo "--- $* $(date -u +%H:%M:%S)"; }  # progress only: GPUs idle during prep by design

step "START node prep"
NODE_PREP="[ -f ~/.rdsp_node_prep_done ] || bash ~/rdsp/bench/qwen35_35b/node_prep.sh"
$SSH2 "$NODE_PREP" < /dev/null >> "$LOGS/node2_prep.log" 2>&1 &
N2=$!
bash -c "$NODE_PREP" >> "$LOGS/node1_prep.log" 2>&1; rc1=$?
wait $N2; rc2=$?
cp ~/hardware.txt "$LOGS/hardware-node1.txt"; $SSH2 "cat ~/hardware.txt" < /dev/null > "$LOGS/hardware-node2.txt"
[ $rc1 = 0 ] && [ $rc2 = 0 ] || { mark "PREP FAILED node prep: node 1 exit $rc1, node 2 exit $rc2"; exit 1; }
step "END node prep"

step "START convert and export"
CONTAINER="sudo docker run --gpus all --rm --ipc=host --shm-size=64g --ulimit memlock=-1 \
  -v $HOME/hf:/root/.cache/huggingface -v $HOME/mimo:/workspace -v $RDSP:/rdsp \
  -e HF_HOME=/root/.cache/huggingface -e PYTHONUNBUFFERED=1 -w /opt/Megatron-Bridge nvcr.io/nvidia/nemo:26.08"
if [ -z "${SKIP_CONVERT:-}" ] && [ ! -f "$HOME/mimo/std/$MODEL.done" ]; then
  $CONTAINER bash -c "python scripts/conversion/setup_conversion.py import --hf-model Qwen/$MODEL \
    --megatron-path /workspace/std/$MODEL" > "$LOGS/convert.log" 2>&1; rc=$?
  sudo chown -R "$(id -u):$(id -g)" "$HOME/hf" "$HOME/mimo"
  [ $rc = 0 ] || { mark "PREP FAILED convert exit $rc"; exit 1; }
  touch "$HOME/mimo/std/$MODEL.done"
fi
if [ ! -f "$HOME/mimo/cord_steps-$MODEL/stats.json" ]; then
  env LOGS="$LOGS" bash "$RUNS" export > /dev/null 2>&1; rc=$?   # GBS=128, STEPS=50 from two_node.sh
  [ $rc = 0 ] && [ -f "$HOME/mimo/cord_steps-$MODEL/stats.json" ] || { mark "PREP FAILED export exit $rc"; exit 1; }
fi
$SSH2 "mkdir -p ~/mimo/std" < /dev/null   # rsync makes only the last directory of a path
for d in $([ -d "$HOME/mimo/std/$MODEL" ] && echo "std/$MODEL") "cord_steps-$MODEL"; do
  rsync -a -e "ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no" "$HOME/mimo/$d/" "ubuntu@$PRIV2:mimo/$d/" \
    || { mark "PREP FAILED copy $d to node 2"; exit 1; }
done
step "END convert and export"

step "START nccl check"
bash "$BOTH" bash "$RDSP/bench/cluster/nccl_check.sh" > "$LOGS/nccl_check.log" 2>&1
grep -E "bus bandwidth" "$LOGS/nccl_check.log"
grep -m2 -oE "NET/OFI Selected [^ ]+ [^ ]+|Using network [A-Za-z]+|NET/Socket" "$LOGS/nccl_check.log" | sort -u
step "END nccl check"

bash "$RDSP/bench/cluster/ray_cluster.sh" up "$PRIV1" "$PRIV2" > "$LOGS/ray_up.log" 2>&1 \
  || { mark "PREP FAILED ray cluster"; cat "$LOGS/ray_up.log"; exit 1; }
mark "PREP DONE"
