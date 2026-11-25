#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# run2.sh <fit|run> <CONFIG>: TorchTitan over two nodes, Qwen3.5-35B-A3B on Bridge's rows exported
# at 128 per step. Start it on both nodes at once from ~/rdsp with bench/two_node/both.sh, which
# sets NNODES, NODE_RANK and MASTER_ADDR:
#   bash bench/two_node/both.sh bash bench/torchtitan/scripts/run2.sh fit qwen35_35b_a3b_cord_fsdp16
#   fit   3 steps (memory check)        run   50 steps (STEPS to change)
# Each node logs its local rank 0 (global ranks 0 and 8) to $LOGS/<step>-<CONFIG>-node<rank>.log
# (default ../logs).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LOGS=${LOGS:-$HERE/../logs}; mkdir -p "$LOGS"   # LOGS: a layout's own logs directory
CONFIG=${2:?usage: run2.sh <fit|run> <CONFIG>}
case "${1:-}" in
  fit) STEPS=3 ;;
  run) STEPS=${STEPS:-50} ;;
  *) sed -n 2,8p "$0"; exit 1 ;;
esac
# MODEL (default Qwen3.5-35B-A3B), EXPORT_NAME (its rows in ~/mimo, default cord_steps-$MODEL)
MODEL=${MODEL:-Qwen3.5-35B-A3B}
HF_ASSETS=$(ls -d "$HOME"/hf/hub/models--Qwen--$MODEL/snapshots/* | head -1)
# NSYS_DELAY + NSYS_DURATION (seconds): Nsight Systems on node 1's 8 ranks over that window
# (CUDA kernels and NVTX), report $LOGS/nsys-<CONFIG>-node0.nsys-rep
PROFILER=""
if [ -n "${NSYS_DELAY:-}" ] && [ "$NODE_RANK" = 0 ]; then
  PROFILER="nsys profile -t cuda,nvtx -s none --cpuctxsw=none --delay $NSYS_DELAY \
    --duration ${NSYS_DURATION:-36} --force-overwrite true -o $LOGS/nsys-$CONFIG-node0"
fi
cd ~/torchtitan
# NET=socket: NCCL between the nodes over TCP (NCCL_NET=Socket). NAME: a log-name suffix.
(unset LD_LIBRARY_PATH && . ~/tt_venv/bin/activate && . "$HERE/../../../bench/two_node/efa_env.sh" && \
 { [ "${NET:-efa}" != socket ] || export NCCL_NET=Socket; } && \
 env PYTHONPATH="$HERE/.." HF_ASSETS="$HF_ASSETS" EXPORTED="$HOME/mimo/${EXPORT_NAME:-cord_steps-$MODEL}" \
     ROWS=${ROWS:-128} SEQ=${SEQ:-2048} STEPS=$STEPS PYTORCH_ALLOC_CONF="expandable_segments:True" LOG_RANK=0 \
     $PROFILER torchrun --nnodes "$NNODES" --node_rank "$NODE_RANK" --master_addr "$MASTER_ADDR" \
       --master_port 29413 --nproc_per_node 8 --local-ranks-filter 0 --role rank --tee 3 \
       -m torchtitan.train --module cord_qwen35 --config "$CONFIG") \
  2>&1 | tee "$LOGS/$1-$CONFIG${NAME:-}-node$NODE_RANK.log"
