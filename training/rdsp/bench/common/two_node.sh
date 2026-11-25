# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# Sourced by the two-node scripts on node 1 (bench/qwen35_35b/*.sh, bench/mimo/rdsp_27b.sh), after
# bench/qwen35_35b/prep.sh: Qwen3.5-35B-A3B on two p5.48xlarge (16 H100, EFA), the settings every
# run shares. Bridge and TorchTitan start on both nodes through bench/cluster/both.sh; rdsp's
# driver runs here, its actors on the Ray cluster bench/cluster/ray_cluster.sh runs over both nodes.
COMMON=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RDSP=$(cd "$COMMON/../.." && pwd)
# MODEL=Qwen3.5-27B: the dense 27B on the same layouts (its logs in logs/<layout>-27b/)
export MODEL=${MODEL:-Qwen3.5-35B-A3B} GBS=${GBS:-128}
case $MODEL in
  Qwen3.5-27B) export RECIPE=qwen35_vl_27b_sft_16gpu_h100_bf16_config LOGS_SUFFIX=-27b ;;
  *)           export RECIPE=qwen35_vl_35b_a3b_sft_16gpu_h100_bf16_config LOGS_SUFFIX="" ;;
esac
export STEPS=${STEPS:-50}
RUNS=$COMMON/runs.sh
BOTH=$RDSP/bench/cluster/both.sh
DATA=exported:$HOME/mimo/${EXPORT_NAME:-cord_steps-$MODEL}   # EXPORT_NAME: other rows
# Bridge: bf16 gradients with the fp32 master as 2-byte remainders (the 27B runs' settings), no
# router load-balancing loss, every row padded to 2048 (pipelines pad on their own anyway), and
# a 60-minute NCCL timeout: Bridge's first step compiles for 10-18 minutes.
MEG_COMMON="$(cat "$COMMON/meg_extra_bf16.txt") \
model.moe_router_load_balancing_type=none model.moe_aux_loss_coeff=0.0 \
dataset.pad_to_max_length=true dist.distributed_timeout_minutes=60"
# Megatron refuses sequence parallelism (the recipe's, for its TP2) without tensor parallelism
MEG_TP1="model.sequence_parallel=false"
MEG_RC="model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1"
# rdsp: the 27B runs' flags (bench/qwen35_27b/rdsp_pp8.sh) with the 2-byte
# master. MOE_EXPERTS=hf (default): Hugging Face's grouped matmuls for the experts. Liger's fused
# kernel (liger) took 350 s for the first step against 24 s (2026-10-18, PP8xDP2), compiling and
# benchmarking its kernels; a stage waiting that long hits NCCL's 5-minute timeout.
RDSP_COMMON="--seq ${SEQ:-2048} --pad-multiple 0 --lr 1e-5 --betas 0.9,0.95 --eps 1e-8 --weight-decay 0.0 \
--grad-dtype bf16 --master-dtype remainder --fused-adam --optimizer-group-numel 250000000 \
--loss token-mean --liger \
--moe-experts ${MOE_EXPERTS:-hf} --prefetch --sharded-loss --drop-padding-mask"
# RECOMPUTE=1: every block recomputed, on whichever system runs (a run that does not fit without)
RECOMPUTE=${RECOMPUTE:-0}
# NET=socket: NCCL between the nodes over TCP instead of EFA (NCCL_NET=Socket; NVLink inside a
# node as before), as on a cluster without EFA or InfiniBand. Logs get "-tcp" after NAME.
export NET=${NET:-efa}
[ "$NET" = socket ] && export NAME="${NAME:-}-tcp"
# ~/no_tcp: TCP runs switched off (a script already running skips them when it gets there)
if [ "$NET" = socket ] && [ -f "$HOME/no_tcp" ]; then echo "TCP runs off (~/no_tcp)"; exit 0; fi

bridge() {  # bridge <runs.sh case> <log name> [Bridge overrides...]: Bridge on both nodes
  local case=$1 name=$2 rc=0; shift 2
  local rc_flags=""; [ "$RECOMPUTE" = 1 ] && rc_flags=$MEG_RC
  bash "$BOTH" env MODEL=$MODEL RECIPE=$RECIPE GBS=$GBS STEPS=$STEPS LOGS=$LOGS NET=$NET \
    NCCL_DEBUG=${NCCL_DEBUG:-} HOST_OFI=${HOST_OFI:-} NCCL_P2P_PXN_LEVEL=${NCCL_P2P_PXN_LEVEL:-} \
    GRAD_REDUCE_FP32=false MEGATRON_EXTRA="$MEG_COMMON $rc_flags $*" bash "$RUNS" "$case" || rc=$?
  mv "$LOGS/$case.log" "$LOGS/$name.log" 2>/dev/null || true
  return $rc
}

rdsp() {  # rdsp <log name> [train_vl.py layout flags...]: rdsp, actors on both nodes
  local name=$1; shift
  (unset LD_LIBRARY_PATH && . "$HOME/venv/bin/activate" && . "$HOME/cuda_env.sh" && \
   . "$RDSP/bench/cluster/efa_env.sh" && \
   { [ "$NET" != socket ] || export NCCL_NET=Socket; } && \
   env DS_BUILD_OPS=0 HF_HOME=$HOME/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       RAY_ADDRESS=auto python "$RDSP/train_vl.py" --model "Qwen/$MODEL" --dataset "$DATA" \
       --steps "$STEPS" $RDSP_COMMON "$@" ${RDSP_ARGS:-}) 2>&1 | tee "$LOGS/$name.log"
  return "${PIPESTATUS[0]}"
}

vision_first() {  # vision_first <gpus>: stage 0's --stage flag where it holds the vision encoder.
  # Its encoder is recomputed (recompute_blocks=0: the encoder's blocks, none of the decoder's):
  # without, stage 0 ran out of memory when Adam's state appeared at step 1 (PP8xDP2, 76 GB of
  # 79 in use). VISION_RC=0 keeps the activations; RECOMPUTE=1 recomputes every block anyway.
  local rc=",recompute_blocks=0"
  [ "${VISION_RC:-1}" = 0 ] && rc=""
  [ "$RECOMPUTE" = 1 ] && rc=",recompute=1"
  echo "--stage 0:gpus=$1,tp=1,zero=1$rc"
}

stage_flags() {  # stage_flags <first stage> <stages> <gpus each> [extra]: --stage flags
  local s out="" rc=""; [ "$RECOMPUTE" = 1 ] && rc=",recompute=1"
  for ((s = $1; s < $1 + $2; s++)); do out="$out --stage $s:gpus=$3,tp=1,zero=1$rc${4:-}"; done
  echo "$out"
}
