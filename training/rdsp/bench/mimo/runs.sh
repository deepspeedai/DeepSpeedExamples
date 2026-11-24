#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# rdsp vs Megatron (MegatronMIMO and Megatron-Bridge) on one 8-GPU node,
# Qwen3.5 on CORD-v2. Run on the node after bench/setup_node.sh:
#
#   MODEL=Qwen3.5-2B runs.sh convert             # HF -> Megatron checkpoints
#   MODEL=Qwen3.5-2B runs.sh export              # Bridge's samples, replayed by rdsp
#   MODEL=Qwen3.5-2B runs.sh noncoloc-mimo  | noncoloc-rdsp   # vision 1 GPU + language TP2xDP2
#   MODEL=Qwen3.5-2B runs.sh shared-megatron | shared-rdsp    # TP2xPP2xDP2
#   MODEL=Qwen3.5-4B runs.sh tp4pp2-megatron | tp4pp2-rdsp    # TP4xPP2xDP1
#   MODEL=Qwen3.5-4B runs.sh coloc-rdsp      # rdsp only: colocated vision
#   vision2-* (images DP2), langpp2-* (language TP2xPP2), langtp4-* (language TP4):
#   more non-colocated MIMO layouts, each as <case>-mimo | <case>-rdsp
#   python summarize.py ~/runs                   # the results table
#
# Case names are the same for both models: set LOGS=~/runs-4b for the 4B runs.
#
# GBS: rows per step (default 64; the two-node runs use 128). EXPORT_NAME: the export's folder in
# ~/mimo (default cord_steps-$MODEL).
# Settings held equal: seq 2048, 64 rows per step, one row per GPU per
# microbatch, AdamW lr 1e-5 constant, betas 0.9/0.95, eps 1e-8, no weight
# decay, no clipping, no warm-up, bf16 with fp32 gradients and optimizer
# state split over data parallel, no recompute, 50 steps. Megatron's
# defaults that differ are overridden here (clipping, warm-up, cosine decay,
# the MTP head, the shuffling sampler). Loss: the Bridge runs train on the
# token mean (calculate_per_token_loss); MegatronMIMO trains on per-
# microbatch means and the rdsp run against it does the same. rdsp runs the
# recipes in ../../recipes with DATA=exported:<Bridge's samples>.
set -euo pipefail
MODEL=${MODEL:-Qwen3.5-2B}
export STEPS=${STEPS:-50}
WORK=$HOME/mimo                       # /workspace inside the container
IMAGE=nvcr.io/nvidia/nemo:26.08
LOGS=${LOGS:-$HOME/runs}; mkdir -p "$LOGS"
if [ -n "${NSYS_DELAY:-}" ]; then mkdir -p "$WORK/nsys"; fi
RDSP=$(cd "$(dirname "$0")/../.." && pwd)
SIZE=$(echo "${MODEL#Qwen3.5-}" | tr '[:upper:]' '[:lower:]')    # 2b, 4b
case $SIZE in
  2b) RECIPE=${RECIPE:-qwen35_vl_2b_sft_1gpu_h100_bf16_config} ;;
  *)  RECIPE=${RECIPE:-qwen35_vl_${SIZE}_sft_2gpu_h100_bf16_config} ;;
esac

# Two nodes (NNODES=2, NODE_RANK, MASTER_ADDR): host networking and every device (EFA), NCCL
# over EFA through the container's own aws-ofi-nccl plugin (its ld.so.conf lists it; replacing
# the container's LD_LIBRARY_PATH would also drop its CUDA forward-compatibility library, which
# its CUDA 13.3 needs on a 13.2 driver); NET=socket forces TCP instead.
NNODES=${NNODES:-1}
MULTI=""
if [ "$NNODES" -gt 1 ]; then
  MULTI="--network host --privileged -e FI_PROVIDER=efa -e FI_EFA_USE_DEVICE_RDMA=1 \
    -e NCCL_SOCKET_IFNAME=^lo,docker"
  [ "${NET:-efa}" = socket ] && MULTI="$MULTI -e NCCL_NET=Socket"
  # HOST_OFI=1: the host's aws-ofi-nccl (1.21.1, with its Libfabric) mounted over the container's
  # (1.17.3, built for NCCL 2.28, beside the container's NCCL 2.30.7). NCCL_P2P_PXN_LEVEL: passed
  # through (0: each rank's send/recv on its own NIC, not through another GPU's process). Both for
  # Bridge's pipeline across the nodes, which deadlocks in batched send/recv.
  [ -n "${HOST_OFI:-}" ] && MULTI="$MULTI -v /opt/amazon/ofi-nccl:/opt/amazon/ofi-nccl:ro \
    -v /opt/amazon/efa:/opt/amazon/efa:ro"
  [ -n "${NCCL_P2P_PXN_LEVEL:-}" ] && MULTI="$MULTI -e NCCL_P2P_PXN_LEVEL=$NCCL_P2P_PXN_LEVEL"
fi
# Compiled Triton and TorchInductor kernels persist in $WORK/.cache across containers: a run's
# first step compiles for many minutes (35B on two nodes: 30+), the next run reuses them. Steps
# after the first, which the results use, are unchanged.
container() {  # container <command...>
  sudo docker run --gpus all --rm --ipc=host --shm-size=64g --ulimit memlock=-1 $MULTI \
    -v "$HOME/hf:/root/.cache/huggingface" -v "$WORK:/workspace" -v "$RDSP:/rdsp" \
    -e HF_HOME=/root/.cache/huggingface -e PYTHONUNBUFFERED=1 -w /opt/Megatron-Bridge \
    -e TRITON_CACHE_DIR=/workspace/.cache/triton -e TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/inductor \
    ${NCCL_DEBUG:+-e NCCL_DEBUG=$NCCL_DEBUG} \
    "$IMAGE" bash -c "$*"
  sudo chown -R "$(id -u):$(id -g)" "$HOME/hf" "$WORK"   # the container writes as root
}

# SEQ (default 2048) and DATASET (a Bridge preset, default cord_v2): NVIDIA's published SFT run
# uses DATASET=medpix SEQ=4096 GBS=32
STD_DATA="model.seq_length=${SEQ:-2048} dataset.seq_length=${SEQ:-2048} dataset.source.dataset_name=${DATASET:-cord_v2} \
  dataset.dataloader_type=single train.global_batch_size=${GBS:-64}"
STD_TRAIN="$STD_DATA train.micro_batch_size=1 train.train_iters=$STEPS \
  checkpoint.pretrained_checkpoint=/workspace/std/$MODEL checkpoint.save=null \
  model.mtp_num_layers=null model.calculate_per_token_loss=true \
  model.recompute_granularity=null \
  optimizer.lr=1e-5 optimizer.min_lr=1e-5 optimizer.adam_beta1=0.9 optimizer.adam_beta2=0.95 \
  optimizer.adam_eps=1e-8 optimizer.weight_decay=0.0 optimizer.clip_grad=0.0 \
  scheduler.lr_warmup_iters=0 scheduler.lr_decay_style=constant \
  scheduler.start_weight_decay=0.0 scheduler.end_weight_decay=0.0 \
  ddp.grad_reduce_in_fp32=${GRAD_REDUCE_FP32:-true} ddp.use_distributed_optimizer=true ddp.average_in_collective=false \
  validation.eval_iters=0 logger.log_interval=1"
# Diagnosis only, never for published runs: MIMO_EXTRA (e.g. "--pad-to-seq-length
# false"), MEGATRON_EXTRA (Bridge overrides, e.g. "logger.timing_log_level=2"),
# RDSP_EXTRA (train_vl.py flags, e.g. "--profile"), NCCL_DEBUG=INFO.
# GRAD_REDUCE_FP32=false: Bridge's gradient buffer in bf16 (only with mixed_precision=null
# in MEGATRON_EXTRA; Bridge's bf16_mixed resets it to true).
# NSYS_DELAY + NSYS_DURATION (seconds): Nsight Systems on that window of the
# run, every process, report in ~/mimo/nsys/<case>.nsys-rep. NSYS_FLAGS replaces the
# default "-s none --cpuctxsw=none" (e.g. call stacks for every sync).
MIMO_TRAIN="--hf-model Qwen/$MODEL --dataset-name cord_v2 --seq-length 2048 \
  --global-batch-size ${GBS:-64} --train-iters $STEPS --dataloader-type single \
  --pretrained-checkpoint /workspace/$MODEL-mimo \
  --lr 1e-5 --min-lr 1e-5 --adam-beta1 0.9 --adam-beta2 0.95 --weight-decay 0.0 \
  --start-weight-decay 0.0 --end-weight-decay 0.0 --clip-grad 0.0 --lr-warmup-iters 0 \
  --log-interval 1 --experiment-root /workspace/runs ${MIMO_EXTRA:-}"

nsys_prefix() {  # nsys_prefix <report path>: a profiler prefix when NSYS_DELAY and NSYS_DURATION are set
  [ -n "${NSYS_DELAY:-}" ] && [ -n "${NSYS_DURATION:-}" ] || return 0
  echo "nsys profile -t cuda,nvtx ${NSYS_FLAGS:--s none --cpuctxsw=none} --delay $NSYS_DELAY" \
       "--duration $NSYS_DURATION --force-overwrite true -o $1"
}

megatron() {  # megatron <log name> <tp> <pp>: the standard Bridge recipe
  container "$(nsys_prefix "/workspace/nsys/$1") python -m torch.distributed.run --nproc_per_node=8 \
    --nnodes $NNODES --node_rank ${NODE_RANK:-0} --master_addr ${MASTER_ADDR:-127.0.0.1} --master_port 29411 \
    scripts/training/run_recipe.py \
    --recipe $RECIPE --step_func qwen3_vl_step $STD_TRAIN \
    model.tensor_model_parallel_size=$2 model.pipeline_model_parallel_size=$3 ${MEGATRON_EXTRA:-}" \
    2>&1 | tee "$LOGS/$1.log"
}

mimo() {  # mimo <log name> <GPUs> <global microbatch> <language layout> <images layout>
  # MIMO_PATCH=1: a /tmp copy of MIMO's launcher with Bridge's memory settings
  # (bench/gap_round/mimo/patch_mimo.py; the container's own file stays as shipped);
  # MIMO_RECOMPUTE=1 then also recomputes every language layer
  local script=examples/megatron_mimo/qwen35_vl/finetune_qwen35_vl.py prep=""
  if [ -n "${MIMO_PATCH:-}" ]; then
    prep="cp $script /tmp/mimo_mem.py && python /rdsp/bench/gap_round/mimo/patch_mimo.py /tmp/mimo_mem.py &&"
    script=/tmp/mimo_mem.py
  fi
  # MIMO_NPROC0 / 1 / 2: processes on node 1 / node 2 / a third box when they differ (e.g. language on node
  # 1's 8 GPUs, the vision encoder alone on 1 GPU of node 2: 8 and 1); default GPUs / NNODES each
  local nproc=$(( $2 / NNODES ))
  [ "${NODE_RANK:-0}" = 0 ] && nproc=${MIMO_NPROC0:-$nproc}
  [ "${NODE_RANK:-0}" = 1 ] && nproc=${MIMO_NPROC1:-$nproc}
  [ "${NODE_RANK:-0}" = 2 ] && nproc=${MIMO_NPROC2:-$nproc}
  container "$prep MIMO_RECOMPUTE=${MIMO_RECOMPUTE:-0} $(nsys_prefix "/workspace/nsys/$1") python -m torch.distributed.run --nproc_per_node=$nproc \
    --nnodes $NNODES --node_rank ${NODE_RANK:-0} --master_addr ${MASTER_ADDR:-127.0.0.1} --master_port 29412 \
    $script $MIMO_TRAIN --micro-batch-size $3 \
    --run-name $1 --component language=$4 --component images=$5" 2>&1 | tee "$LOGS/$1.log"
}

rdsp() {  # rdsp <log name> <recipe> [train_vl flags]
  local name=$1 recipe=$RDSP/recipes/$2; shift 2
  [ -f "$recipe" ] || { echo "no recipe $recipe for $MODEL" >&2; exit 1; }
  (unset LD_LIBRARY_PATH && . "$HOME/venv/bin/activate" && . "$HOME/cuda_env.sh" && \
   DS_BUILD_OPS=0 HF_HOME=$HOME/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   DATA=exported:$WORK/cord_steps-$MODEL $(nsys_prefix "$WORK/nsys/$name") \
   bash "$recipe" --steps "$STEPS" "$@" ${RDSP_EXTRA:-}) \
    2>&1 | tee "$LOGS/$name.log"
}

case "${1:-}" in
  convert)
    container "python scripts/conversion/setup_conversion.py import --hf-model Qwen/$MODEL \
        --megatron-path /workspace/std/$MODEL && \
      python -m torch.distributed.run --nproc_per_node=2 examples/conversion/convert_megatron_mimo.py \
        import --hf-model Qwen/$MODEL --megatron-path /workspace/$MODEL-mimo \
        --component language=tp=1,dp=1,rank_offset=0 --component images=tp=1,dp=1,rank_offset=1 \
        --torch-dtype bfloat16" 2>&1 | tee "$LOGS/convert.log" ;;
  export)
    container "python /rdsp/bench/mimo/export_bridge_batches.py --out /workspace/${EXPORT_NAME:-cord_steps-$MODEL} \
      --steps $STEPS --recipe $RECIPE $STD_DATA dataset.pad_to_max_length=true" \
      2>&1 | tee "$LOGS/export.log" ;;
  # MegatronMIMO's microbatch is global over language DP: 1 row per language rank
  noncoloc-mimo)     # language TP2 x DP2 on ranks 0-3, images on rank 4
    mimo noncoloc-mimo 5 2 tp=2,pp=1,dp=2,rank_offset=0 tp=1,pp=1,dp=1,rank_offset=4 ;;
  noncoloc-rdsp)
    rdsp noncoloc-rdsp "qwen35-$SIZE-vision1-tp2dp2.sh" ;;
  vision2-mimo)      # language TP2 x DP2 on ranks 0-3, images DP2 on ranks 4-5
    mimo vision2-mimo 6 2 tp=2,pp=1,dp=2,rank_offset=0 tp=1,pp=1,dp=2,rank_offset=4 ;;
  vision2-rdsp)
    rdsp vision2-rdsp "qwen35-$SIZE-vision2-tp2dp2.sh" ;;
  mimo-layout)       # any grid: MIMO_NAME MIMO_GPUS MIMO_MBS MIMO_LANGUAGE MIMO_IMAGES from the caller
    mimo "${MIMO_NAME:?}" "${MIMO_GPUS:?}" "${MIMO_MBS:?}" "${MIMO_LANGUAGE:?}" "${MIMO_IMAGES:?}" ;;
  langpp2-mimo)      # language TP2 x PP2 on ranks 0-3, images on rank 4
    mimo langpp2-mimo 5 1 tp=2,pp=2,dp=1,rank_offset=0 tp=1,pp=1,dp=1,rank_offset=4 ;;
  langpp2-rdsp)
    rdsp langpp2-rdsp "qwen35-$SIZE-vision1-tp2pp2.sh" ;;
  langtp4-mimo)      # language TP4 on ranks 0-3, images on rank 4
    mimo langtp4-mimo 5 1 tp=4,pp=1,dp=1,rank_offset=0 tp=1,pp=1,dp=1,rank_offset=4 ;;
  langtp4-rdsp)
    rdsp langtp4-rdsp "qwen35-$SIZE-vision1-tp4.sh" ;;
  shared-megatron)
    megatron shared-megatron 2 2 ;;
  shared-rdsp)
    rdsp shared-rdsp "qwen35-$SIZE-tp2pp2dp2.sh" ;;
  tp4pp2-megatron)
    megatron tp4pp2-megatron 4 2 ;;
  tp2dp4-megatron)    # TP2 x DP4, no pipeline
    megatron tp2dp4-megatron 2 1 ;;
  tp4dp2-megatron)    # TP4 x DP2, no pipeline
    megatron tp4dp2-megatron 4 1 ;;
  tp2pp4-megatron)    # TP2 x PP4, no data parallelism
    megatron tp2pp4-megatron 2 4 ;;
  tp1pp8-megatron)    # TP1 x PP8: one GPU per pipeline stage
    megatron tp1pp8-megatron 1 8 ;;
  dp8-megatron)       # data parallel 8, no TP or PP (with Megatron FSDP overrides: fully sharded)
    megatron dp8-megatron 1 1 ;;
  twonode-mimo)       # two nodes: language TP4 x PP2 on node 1 (ranks 0-7), images DP8 on node 2
    mimo twonode-mimo 16 8 tp=4,pp=2,dp=1,rank_offset=0 tp=1,pp=1,dp=8,rank_offset=8 ;;
  convert-mimo)       # only the MegatronMIMO checkpoint: language on GPUs 0..CONVERT_TP-1 (TP
    # CONVERT_TP, default 1; the 27B needs 4: on one GPU its conversion runs out of memory), images
    # on the next GPU. Training reshards it on load.
    ctp=${CONVERT_TP:-1}
    container "python -m torch.distributed.run --nproc_per_node=$((ctp + 1)) examples/conversion/convert_megatron_mimo.py \
        import --hf-model Qwen/$MODEL --megatron-path /workspace/$MODEL-mimo \
        --component language=tp=$ctp,dp=1,rank_offset=0 --component images=tp=1,dp=1,rank_offset=$ctp \
        --torch-dtype bfloat16" 2>&1 | tee "$LOGS/convert-mimo.log" ;;
  tp2ep16-megatron)   # two nodes: TP2 x DP8, no pipeline; the experts' split comes with the
    megatron tp2ep16-megatron 2 1 ;;   # overrides (model.expert_model_parallel_size=16)
  custom-megatron)    # any TP x PP (DP from the GPU count), log $LOGS/<CUSTOM_NAME>.log; experts,
    # layout and the rest come with MEGATRON_EXTRA
    megatron "${CUSTOM_NAME:?}" "${CUSTOM_TP:?}" "${CUSTOM_PP:?}" ;;
  nccl-check)         # bench/two_node/nccl_check.py inside the container (its NCCL and EFA plugin),
    # every GPU of NNODES nodes; NCCL_DEBUG=INFO shows the network it picked
    container "NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET python -m torch.distributed.run \
      --nproc_per_node=8 --nnodes $NNODES --node_rank ${NODE_RANK:-0} \
      --master_addr ${MASTER_ADDR:-127.0.0.1} --master_port 29415 /rdsp/bench/two_node/nccl_check.py" \
      2>&1 | tee "$LOGS/nccl-check-container.log" ;;
  tp4pp2-rdsp)
    rdsp tp4pp2-rdsp "qwen35-$SIZE-tp4pp2dp1.sh" ;;
  coloc-rdsp)        # rdsp only: vision encoder on every GPU, language TP2 x PP2 x DP2
    rdsp coloc-rdsp "qwen35-$SIZE-coloc-tp2pp2dp2.sh" ;;
  *) sed -n 2,15p "$0"; exit 1 ;;
esac
