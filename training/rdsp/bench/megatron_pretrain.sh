#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# Megatron-LM pretrain_gpt.py, Qwen3-0.6B shape, mock data. One benchmark cell.
#   PP=2 M=8 SEQ=2048 VARIANT=plain ./megatron_pretrain.sh
#   PP=4 M=16 SEQ=512 VARIANT=vpp   ./megatron_pretrain.sh
# VARIANT: plain    = optional fusions off, bf16 grad summing (matched to unfused rdsp)
#          defaults = Megatron's standard fusions
#          vpp      = defaults + interleaved (virtual) pipeline stages
#          local    = Megatron's non-TransformerEngine layers
set -euo pipefail
PP=${PP:-2}; M=${M:-8}; SEQ=${SEQ:-2048}; VARIANT=${VARIANT:-plain}
ITERS=${ITERS:-60}; TIMING=${TIMING:-0}           # TIMING=2 for instrumented runs
MEGATRON=${MEGATRON:-/opt/Megatron-LM}
MBS=$(( SEQ == 512 ? 4 : 1 )); GBS=$(( MBS * M ))  # 2048 tokens per microbatch

MODEL=(
  --num-layers 28 --hidden-size 1024 --ffn-hidden-size 3072
  --num-attention-heads 16 --group-query-attention --num-query-groups 8
  --kv-channels 128 --qk-layernorm
  --normalization RMSNorm --norm-epsilon 1e-6 --swiglu --disable-bias-linear
  --position-embedding-type rope --rotary-base 1000000 --rotary-percent 1.0
  --untie-embeddings-and-output-weights --max-position-embeddings 40960
  --attention-dropout 0.0 --hidden-dropout 0.0 --init-method-std 0.02
  --seq-length "$SEQ"
  --vocab-size 151936 --make-vocab-size-divisible-by 128
  --tokenizer-type NullTokenizer --mock-data
)
TRAIN=(
  --micro-batch-size "$MBS" --global-batch-size "$GBS" --train-iters "$ITERS"
  --optimizer adam --lr 1e-5 --min-lr 1e-5 --lr-decay-style constant --lr-warmup-iters 0
  --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8
  --clip-grad 0.0 --bf16 --seed 1234
  --eval-iters 0 --eval-interval "$ITERS" --log-interval 1 --log-throughput
  --timing-log-level "$TIMING" --timing-log-option all
)
PAR=( --tensor-model-parallel-size 1 --pipeline-model-parallel-size "$PP" --context-parallel-size 1 )

case "$VARIANT" in
  plain)
    IMPL=( --transformer-impl transformer_engine --attention-backend flash
           --no-gradient-accumulation-fusion --no-bias-swiglu-fusion --no-rope-fusion
           --no-bias-dropout-fusion --no-persist-layer-norm --no-masked-softmax-fusion
           --grad-reduce-in-bf16 --no-overlap-p2p-communication ) ;;
  defaults|vpp)
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    IMPL=( --transformer-impl transformer_engine --attention-backend auto
           --cross-entropy-loss-fusion --manual-gc )
    if [[ "$VARIANT" == vpp ]]; then
      if [[ "$PP" == 2 ]]; then IMPL+=( --num-virtual-stages-per-pipeline-rank 2 )
      else IMPL+=( --pipeline-model-parallel-layout "Et*4|t*3|t*3|t*4|t*3|t*4|t*4|t*3,L" ); fi
    fi ;;
  local)
    IMPL=( --transformer-impl local --no-gradient-accumulation-fusion --grad-reduce-in-bf16 ) ;;
esac
# Explicit layer layout (e.g. PP=8 with 28 layers, which does not divide evenly)
if [[ -n "${LAYOUT:-}" ]]; then
  IMPL+=( --pipeline-model-parallel-layout "$LAYOUT" )
fi
# Balanced split (PP=4): the LM head is ~1/4 of the FLOPs, so the last stage gets 1 layer
if [[ "${BALANCED:-0}" == 1 && "$PP" == 4 ]]; then
  IMPL+=( --pipeline-model-parallel-layout "Et*9|t*9|t*9|t,L" )
fi

cd "$MEGATRON"
exec torchrun --standalone --nproc_per_node "$PP" pretrain_gpt.py \
  "${MODEL[@]}" "${TRAIN[@]}" "${PAR[@]}" "${IMPL[@]}"
# Parse from stdout: "elapsed time per iteration (ms)", "throughput per GPU",
# "lm loss", and at TIMING=2 the per-rank forward-compute/backward-compute/*-send/*-recv.
