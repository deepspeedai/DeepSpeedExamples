#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# ray_cluster3.sh <up|down>, on node 1: ray_cluster.sh over three nodes, head on node 1, workers on
# node 2 (~/env2.sh) and the third box (~/env3.sh, which exports PRIV3), every raylet started with
# efa_env.sh. A driver on any of them joins with RAY_ADDRESS=auto; rdsp's stages fill the driver's
# node first.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. ~/env2.sh
. ~/env3.sh
ssh_to() { local host=$1; shift; ssh -i "$HOME/key2.pem" -o StrictHostKeyChecking=no "ubuntu@$host" "$@"; }
START=". ~/venv/bin/activate; . $HERE/efa_env.sh; ray start --disable-usage-stats"
stop() {
  for host in "$PRIV2" "$PRIV3"; do
    ssh_to "$host" ". ~/venv/bin/activate; ray stop --force" < /dev/null > /dev/null 2>&1 || true
  done
  bash -c ". ~/venv/bin/activate; ray stop --force" > /dev/null 2>&1 || true
}
case ${1:-} in
  up)
    stop
    bash -c "$START --num-gpus 8 --head --port 6379 --node-ip-address $PRIV1" > ~/ray_head.log 2>&1
    ssh_to "$PRIV2" "$START --num-gpus 8 --address $PRIV1:6379 --node-ip-address $PRIV2 > ~/ray_worker.log 2>&1" < /dev/null
    ssh_to "$PRIV3" "$START --num-gpus 1 --address $PRIV1:6379 --node-ip-address $PRIV3 > ~/ray_worker.log 2>&1" < /dev/null
    for _ in $(seq 1 30); do   # wait until all 17 GPUs are in the cluster
      n=$(bash -c ". ~/venv/bin/activate; python -c \"import ray; ray.init(address='auto', \
log_to_driver=False); print(int(ray.cluster_resources().get('GPU', 0)))\"" 2>/dev/null | tail -1)
      [ "$n" = 17 ] && { echo "ray cluster up: 17 GPUs"; exit 0; }
      sleep 2
    done
    echo "ray cluster has ${n:-0} GPUs, not 17" >&2; exit 1 ;;
  down)
    stop ;;
  *) echo "usage: $0 <up|down>" >&2; exit 2 ;;
esac
