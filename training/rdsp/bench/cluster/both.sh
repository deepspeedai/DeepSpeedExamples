#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# both.sh <command...>, on node 1 from ~/rdsp: the same command on both nodes at once, for
# launchers that start one torchrun per node (Bridge and MIMO through bench/common/runs.sh,
# TorchTitan). It runs with NNODES=2, NODE_RANK (0 here, 1 on node 2) and MASTER_ADDR (node 1)
# in its environment; pass any other setting inside the command (env MODEL=... bash ...), since
# node 2 gets only the command. Node 2 runs it from ~/rdsp too, its output in ~/both-<time>.log
# there. Exits with node 1's code, or 1 if node 1 passed and node 2 failed.
# Fail fast: once either node's command ends, the other gets GRACE seconds (default 120) to end
# too, then its containers and torchrun are killed (otherwise a crash on one node leaves the other
# waiting in NCCL or the rendezvous until the run's own timeout).
# both.sh --cleanup: kill every container and torchrun on both nodes.
set -uo pipefail
. ~/env2.sh
SSH2="ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no -o BatchMode=yes ubuntu@$PRIV2"
KILL="sudo docker ps -q | xargs -r sudo docker kill >/dev/null 2>&1; pkill -f '[t]orch.distributed.run' ; pkill -f '[t]orchrun' ; true"
if [ "${1:-}" = --cleanup ]; then
  $SSH2 "$KILL" < /dev/null; bash -c "$KILL"; exit 0
fi
LOG2=both-$(date -u +%H%M%S).log
$SSH2 "cd ~/rdsp && NNODES=2 NODE_RANK=1 MASTER_ADDR=$PRIV1 $(printf '%q ' "$@") > ~/$LOG2 2>&1" < /dev/null &
PEER=$!
NNODES=2 NODE_RANK=0 MASTER_ADDR=$PRIV1 "$@" &
HERE=$!
while kill -0 $HERE 2>/dev/null && kill -0 $PEER 2>/dev/null; do sleep 5; done
for _ in $(seq 1 $(( ${GRACE:-120} / 5 ))); do
  kill -0 $HERE 2>/dev/null || kill -0 $PEER 2>/dev/null || break
  sleep 5
done
if kill -0 $HERE 2>/dev/null || kill -0 $PEER 2>/dev/null; then
  echo "both.sh: one node ended, the other still running after ${GRACE:-120} s: killing it"
  $SSH2 "$KILL" < /dev/null; bash -c "$KILL"
  sleep 10; kill $HERE $PEER 2>/dev/null   # whatever the kill above did not end
fi
wait $HERE; rc=$?
wait $PEER; rc2=$?
echo "both.sh: node 1 exit $rc, node 2 exit $rc2 (node 2 log ~/$LOG2)"
[ $rc = 0 ] && [ $rc2 != 0 ] && rc=1
exit $rc
