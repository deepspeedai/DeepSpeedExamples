#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# ray_cluster.sh <up|down> <node 1 private IP> <node 2 private IP>, on node 1: the 2-node Ray
# cluster, head on node 1 and a worker on node 2, both raylets started with efa_env.sh so
# every rdsp actor's NCCL uses EFA. A driver on node 1 joins it with RAY_ADDRESS=auto.
# Node 2 is reached with ~/key2.pem.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ACTION=$1 PRIV1=$2 PRIV2=$3
SSH2="ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no ubuntu@$PRIV2"
START=". ~/venv/bin/activate; . $HERE/efa_env.sh; ray start --disable-usage-stats --num-gpus 8"
stop() {
  $SSH2 ". ~/venv/bin/activate; ray stop --force" < /dev/null > /dev/null 2>&1 || true
  bash -c ". ~/venv/bin/activate; ray stop --force" > /dev/null 2>&1 || true
}
case $ACTION in
  up)
    stop   # a cluster left running (an earlier run, by hand) would refuse a new head
    bash -c "$START --head --port 6379 --node-ip-address $PRIV1" > ~/ray_head.log 2>&1
    $SSH2 "$START --address $PRIV1:6379 --node-ip-address $PRIV2 > ~/ray_worker.log 2>&1" < /dev/null
    for _ in $(seq 1 30); do   # wait until both nodes' GPUs are in the cluster
      n=$(bash -c ". ~/venv/bin/activate; python -c \"import ray; ray.init(address='auto', \
log_to_driver=False); print(int(ray.cluster_resources().get('GPU', 0)))\"" 2>/dev/null | tail -1)
      [ "$n" = 16 ] && { echo "ray cluster up: 16 GPUs"; exit 0; }
      sleep 2
    done
    echo "ray cluster has ${n:-0} GPUs, not 16" >&2; exit 1 ;;
  down)
    stop ;;
  *) echo "usage: $0 <up|down> <node 1 private IP> <node 2 private IP>" >&2; exit 2 ;;
esac
