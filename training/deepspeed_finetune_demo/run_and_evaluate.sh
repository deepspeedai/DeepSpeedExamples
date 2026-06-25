#!/bin/bash
# Fine-tune + evaluate: supports MBPP, MMLU, and GSM8K benchmarks
# Usage: bash run_and_evaluate.sh [model_name] [ds_config] [eval_steps] [wandb_name]
#
# Environment variables:
#   BENCHMARK   - one of: mbpp, mmlu, gsm8k (default: mbpp)
#   DATASET     - HuggingFace dataset name (default: auto-selected per benchmark)
#   TP          - tensor parallel size (default: 8)
#   SKIP_TRAIN  - set to 1 to skip training and go straight to eval
#
# Examples:
#   bash run_and_evaluate.sh moonshotai/Moonlight-16B-A3B configs/z2_config.json 100 my_run
#   BENCHMARK=mmlu  bash run_and_evaluate.sh moonshotai/Moonlight-16B-A3B configs/z2_config.json 100
#   BENCHMARK=gsm8k bash run_and_evaluate.sh moonshotai/Moonlight-16B-A3B configs/z2_config.json 100
set -euo pipefail

MODEL=${1:-moonshotai/Moonlight-16B-A3B}
DS_CONFIG=${2:-configs/z2_config.json}
EVAL_STEPS=${3:-100}
WANDB_NAME=${4:-moonlight_finetune}
TP=${TP:-8}
BENCHMARK=${BENCHMARK:-mbpp}

case "$BENCHMARK" in
  mbpp)
    DATASET=${DATASET:-sahil2801/CodeAlpaca-20k}
    ;;
  mmlu)
    DATASET=${DATASET:-cais/mmlu}
    ;;
  gsm8k)
    DATASET=${DATASET:-meta-math/MetaMathQA}
    ;;
  *)
    echo "ERROR: Unknown BENCHMARK '$BENCHMARK'. Use one of: mbpp, mmlu, gsm8k"
    exit 1
    ;;
esac

# Derive a safe directory name from model (replace / with _)
MODEL_SLUG=$(echo "$MODEL" | tr '/' '_')
CONFIG_SLUG=$(basename "$DS_CONFIG" .json)
DATASET_SLUG=$(echo "$DATASET" | tr '/' '_' | tr '[:upper:]' '[:lower:]')

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
PYTHON=${PYTHON:-$(which python3)}
WORKDIR=$(cd "$(dirname "$0")" && pwd)
LOGDIR=$WORKDIR/experiment_logs
OUTPUT_DIR=$WORKDIR/output_${BENCHMARK}_${MODEL_SLUG}_${CONFIG_SLUG}
HF_DIR=$WORKDIR/hf_model_${BENCHMARK}_${MODEL_SLUG}_${CONFIG_SLUG}
BASELINE_DIR=$WORKDIR/eval_results/${BENCHMARK}_baseline_${MODEL_SLUG}
EVAL_DIR=$WORKDIR/eval_results/${BENCHMARK}_${MODEL_SLUG}_${CONFIG_SLUG}

mkdir -p $LOGDIR $BASELINE_DIR $EVAL_DIR

cd $WORKDIR

# ---------- helper: run evaluation by benchmark type ----------
run_eval() {
  local model_path=$1
  local output_dir=$2
  local log_tag=$3
  local gen_log=$LOGDIR/${log_tag}_gen.log
  local eval_log=$LOGDIR/${log_tag}_eval.log

  rm -rf "$output_dir"

  case "$BENCHMARK" in
    mbpp)
      $PYTHON evaluate/humaneval/gen_vllm.py \
        --model "$model_path" \
        --output "$output_dir" \
        --dataset mbpp \
        --tp "$TP" \
        --instruction \
        2>&1 | tee "$gen_log"

      $PYTHON -m evalplus.evaluate \
        --dataset mbpp \
        --samples "$output_dir/samples.jsonl" \
        2>&1 | tee "$eval_log"
      ;;

    mmlu)
      $PYTHON evaluate/mmlu/gen_mmlu.py \
        --model "$model_path" \
        --output "$output_dir" \
        --tp "$TP" \
        2>&1 | tee "$gen_log"

      $PYTHON evaluate/mmlu/eval_mmlu.py \
        --samples "$output_dir/samples.jsonl" \
        2>&1 | tee "$eval_log"
      ;;

    gsm8k)
      $PYTHON evaluate/gsm8k/gen_gsm8k.py \
        --model "$model_path" \
        --output "$output_dir" \
        --tp "$TP" \
        2>&1 | tee "$gen_log"

      $PYTHON evaluate/gsm8k/eval_gsm8k.py \
        --samples "$output_dir/samples.jsonl" \
        2>&1 | tee "$eval_log"
      ;;
  esac
}

# ---------- STEP 0: BASELINE EVALUATION ----------
echo "===== STEP 0: BASELINE EVALUATION (pre-finetune) ====="
echo "Model:      ${MODEL}"
echo "Config:     ${DS_CONFIG}"
echo "Dataset:    ${DATASET}"
echo "Benchmark:  ${BENCHMARK}"
echo "Eval steps: ${EVAL_STEPS}"
echo "W&B name:   ${WANDB_NAME:-<disabled>}"
echo "TP:         ${TP}"
echo "Start: $(date)"

run_eval "$MODEL" "$BASELINE_DIR" "baseline_${BENCHMARK}_${MODEL_SLUG}"
echo "Baseline eval done: $(date)"

if [ "${SKIP_TRAIN:-0}" = "1" ]; then
  echo "SKIP_TRAIN=1: skipping training and convert, jumping to post-finetune eval"
else

# ---------- STEP 1: TRAINING ----------
echo "===== STEP 1: TRAINING ====="
echo "Start: $(date)"
deepspeed --num_gpus=8 finetune_llama.py \
  --model_name $MODEL \
  --output_dir $OUTPUT_DIR \
  --batch_size 16 --max_length 512 \
  --deepspeed_config $DS_CONFIG \
  --dataset_name $DATASET \
  --num_train_epochs 1 \
  --eval_steps $EVAL_STEPS \
  ${WANDB_NAME:+--wandb_name "$WANDB_NAME"} \
  2>&1 | tee $LOGDIR/${BENCHMARK}_${MODEL_SLUG}_${CONFIG_SLUG}_train.log
echo "Training done: $(date)"

# ---------- STEP 2: CONVERT ----------
CKPT_DIR=$(ls -d $OUTPUT_DIR/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
if [ -z "$CKPT_DIR" ]; then
  CKPT_DIR=$OUTPUT_DIR
fi
echo "Using checkpoint: $CKPT_DIR"
$PYTHON convert_ds_to_hf.py \
  --ds_checkpoint $CKPT_DIR \
  --original_model $MODEL \
  --output_dir $HF_DIR \
  --ep_size 8 \
  2>&1 | tee $LOGDIR/${BENCHMARK}_${MODEL_SLUG}_${CONFIG_SLUG}_convert.log
echo "Convert done: $(date)"

# Verify .py files were copied
if [ ! -f "$HF_DIR/modeling_deepseek.py" ]; then
  echo "WARNING: modeling_deepseek.py not found in $HF_DIR"
else
  echo "Verified: custom code files present in HF model dir"
fi

# Delete DS checkpoint to save disk
echo "Removing DS checkpoint to save disk..."
rm -rf $OUTPUT_DIR
echo "DS checkpoint removed"

fi # end SKIP_TRAIN

# ---------- STEP 3: GENERATE + EVALUATE (post-finetune) ----------
echo "===== STEP 3: GENERATE + EVALUATE (post-finetune) ====="
echo "Start: $(date)"
run_eval "$HF_DIR" "$EVAL_DIR" "${BENCHMARK}_${MODEL_SLUG}_${CONFIG_SLUG}"
echo "Evaluate done: $(date)"

# ---------- RESULTS SUMMARY ----------
echo "===== ALL DONE ====="
echo ""
echo "========== RESULTS SUMMARY =========="
echo "--- Baseline (pre-finetune) ---"
grep -E "pass@1|Base|Plus|Accuracy" $LOGDIR/baseline_${BENCHMARK}_${MODEL_SLUG}_eval.log || true
echo ""
echo "--- Finetuned (post-finetune) ---"
grep -E "pass@1|Base|Plus|Accuracy" $LOGDIR/${BENCHMARK}_${MODEL_SLUG}_${CONFIG_SLUG}_eval.log || true
echo "====================================="
