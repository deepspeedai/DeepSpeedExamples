#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# three.sh <command...>, on node 1 from ~/rdsp: both.sh over three nodes, the two in ~/env2.sh and
# the third box in ~/env3.sh (which exports PRIV3, its private IP). The command runs with NNODES=3, NODE_RANK (0
# here, 1 on node 2, 2 on the third box) and MASTER_ADDR (node 1); pass other settings inside the
# command (env VAR=value ...). Node 2's and the third box's output go to ~/three-<time>.log there.
# three.sh --cleanup: kill every container and torchrun on all three.
set -uo pipefail
. ~/env2.sh
. ~/env3.sh
ssh_to() { local host=$1; shift; ssh -i "$HOME/key2.pem" -o StrictHostKeyChecking=no -o BatchMode=yes "ubuntu@$host" "$@"; }
KILL="sudo docker ps -q | xargs -r sudo docker kill >/dev/null 2>&1; pkill -f '[t]orch.distributed.run' ; pkill -f '[t]orchrun' ; true"
if [ "${1:-}" = --cleanup ]; then
  ssh_to "$PRIV2" "$KILL" < /dev/null; ssh_to "$PRIV3" "$KILL" < /dev/null; bash -c "$KILL"; exit 0
fi
LOG=three-$(date -u +%H%M%S).log
CMD=$(printf '%q ' "$@")
ssh_to "$PRIV2" "cd ~/rdsp && NNODES=3 NODE_RANK=1 MASTER_ADDR=$PRIV1 $CMD > ~/$LOG 2>&1" < /dev/null &
P2=$!
ssh_to "$PRIV3" "cd ~/rdsp && NNODES=3 NODE_RANK=2 MASTER_ADDR=$PRIV1 $CMD > ~/$LOG 2>&1" < /dev/null &
P3=$!
NNODES=3 NODE_RANK=0 MASTER_ADDR=$PRIV1 "$@"; rc=$?
wait $P2; rc2=$?
wait $P3; rc3=$?
echo "three.sh: node 1 exit $rc, node 2 exit $rc2, box 3 exit $rc3 (their logs ~/$LOG)"
[ $rc = 0 ] && { [ $rc2 != 0 ] || [ $rc3 != 0 ]; } && rc=1
exit $rc
