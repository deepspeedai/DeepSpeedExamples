#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# one_node.sh <bridge|others>: this node's one-node runs on the dense 27B (8 GPUs). "bridge"
# runs on node 1 while "others" runs on node 2, so the cleanup here touches this node only.
# Each run prints a "=== " line when it starts and when it ends.
#   bridge  Megatron-Bridge TP1 x PP8, every layer recomputed, at three per-stage cuts given
#           with model.pipeline_model_parallel_layout
#   others  TorchTitan FSDP8 (FullAC fp32 reduce, FullAC bf16 reduce, MLP-only recompute
#           bf16 reduce); TorchTitan TP1 x PP8 at rdsp's cut (weights converted to DCP);
#           then rdsp TP1 x PP8 with its measured cut
set -uo pipefail
cd ~/rdsp
B=~/rdsp/bench
P8=$B/qwen35_27b
L8=$P8/logs/pp8
TL=$B/torchtitan/logs
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
cleanup() {
  sudo docker ps -q | xargs -r sudo docker kill > /dev/null 2>&1
  pkill -f "[t]orch.distributed.run"; pkill -f "[t]orchrun"; pkill -f "[t]rain_vl.py"
  bash -c ". ~/venv/bin/activate; ray stop --force" > /dev/null 2>&1
  sleep 10
}
run() {  # run <name> <timeout minutes> <nvidia-smi csv> <command...>
  local name=$1 minutes=$2 csv=$3; shift 3
  mark "START $name"
  $SMI > "$csv" &
  local smi=$!
  timeout $((minutes * 60)) "$@"; local rc=$?
  kill $smi
  cleanup
  mark "END $name exit $rc"
}
# Bridge: every layer recomputed (27B does not fit TP1 x PP8 without); NVIDIA's perf recipes'
# check flags off; 8 data workers (Bridge decodes the images while it trains)
RC_ALL="model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1"
FAST="rerun_state_machine.check_for_nan_in_loss=false ddp.check_for_nan_in_grad=false dataset.num_workers=8"
# The layout reaches Hydra as a quoted string through runs.sh's bash -c
layout() { echo "\"model.pipeline_model_parallel_layout='$1'\""; }
bridge() {  # bridge <name> <layout> [overrides...]
  local name=$1 lay=$2; shift 2
  run "bridge-27b-$name" 60 "$L8/nvsmi-bridge-gap-$name.csv" env NAME="-gap-$name" \
    MEG_RC="$RC_ALL $(layout "$lay") $FAST $*" bash "$P8/pp8.sh" bridge
}
titan() {  # titan <config>
  run "titan-27b-$1" 40 "$TL/nvsmi-gap-$1.csv" env NAME=-gap bash "$B/torchtitan/run.sh" run "$1"
}
mkdir -p "$L8" "$TL"
case "${1:-}" in
  bridge)
    bridge cut1 "Et|t*9|t*10|t*9|t*9|t*9|t*10|t*7L" model.batch_p2p_sync=false
    bridge cut0 "E|t*9|t*10|t*10|t*10|t*9|t*9|t*7L" model.batch_p2p_sync=false
    bridge vp2 "E|t*5|t*5|t*5|t*5|t*5|t*5|t*4|t|t*4|t*5|t*4|t*4|t*4|t*5|t*3L" \
      model.overlap_p2p_comm=true model.batch_p2p_comm=false ;;
  others)
    titan qwen35_27b_cord_fsdp8_rc
    titan qwen35_27b_cord_fsdp8_rc_bf16
    titan qwen35_27b_cord_fsdp8_ffnrc_bf16
    # TorchTitan 8 stages at rdsp's cut: weights converted to DCP first (its HF loader rejects
    # middle pipeline stages)
    HF27=$(ls -d ~/hf/hub/models--Qwen--Qwen3.5-27B/snapshots/* | head -1)
    [ -f ~/tt_dcp/Qwen3.5-27B/.metadata ] || run titan-27b-dcp 40 "$TL/nvsmi-gap-dcp.csv" bash -c \
      "unset LD_LIBRARY_PATH; . ~/tt_venv/bin/activate; cd ~/torchtitan && python \
       scripts/checkpoint_conversion/convert_from_hf.py $HF27 ~/tt_dcp/Qwen3.5-27B \
       --model_name qwen3_5 --model_flavor 27B > $TL/convert-dcp-27b-gap.log 2>&1"
    run titan-27b-pp8 50 "$TL/nvsmi-gap-pp8.csv" env NAME=-gap TT_DCP="$HOME/tt_dcp/Qwen3.5-27B" \
      LOG_RANK=0,7 bash "$B/torchtitan/run.sh" run qwen35_27b_cord_pp8_rc
    run rdsp-27b-pp8 40 "$L8/nvsmi-rdsp-gap.csv" env NAME=-gap \
      RDSP_ARGS="--master-dtype remainder --cuts measured" bash "$P8/pp8.sh" rdsp ;;
  *) sed -n 2,9p "$0"; exit 1 ;;
esac
mark "ONE NODE DONE $1"
