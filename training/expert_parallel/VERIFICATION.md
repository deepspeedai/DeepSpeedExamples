# Qwen3.5 AutoEP Reproduction

This document describes how to reproduce the Qwen3.5 AutoEP sample and compare it with the ZeRO-3 leaf baseline. It is intentionally environment-neutral: choose local output directories and caches that fit your machine or cluster.

The commands below use:

- model preset: `qwen3_5`
- DeepSpeed AutoEP preset: `qwen3_5_moe`
- layers: `8`
- dataset: `wikitext`, `dataset_percentage=10.0`
- tokenizer: `Qwen/Qwen3-0.6B`
- sequence length: `1024`
- micro batch size: `1`
- gradient accumulation: `4`
- world size: `8`
- steps: `100`, with steps `50-99` treated as the post-warmup measurement window

AutoEP uses the sample's built-in `--mode autoep` config with `--autoep_size 8`: bf16, AdamW, ZeRO stage 1, `expert_parallel.enabled=true`, `autoep_size=8`, `preset_model=qwen3_5_moe`. The baseline uses `--mode zero3_leaf`: bf16, AdamW, ZeRO stage 3, and the Qwen3.5 MoE block registered as a ZeRO leaf module.

## Install

Run from the repository root:

```bash
cd training/expert_parallel
python -m pip install -r requirements.txt
export TOKENIZERS_PARALLELISM=false
mkdir -p runs/qwen35/{init,autoep,zero3_leaf,compare}
```

You may set `HF_HOME` and `HF_DATASETS_CACHE` if your environment requires explicit Hugging Face cache locations.


## Qwen3.5 Kernel Requirements

For `--model qwen3_5`, `flash-linear-attention`, `causal-conv1d`, `flash-attn`, and `tilelang` on H100/Triton `>= 3.4` are verification requirements, not optional accelerators. The verification should fail if `transformers.utils.import_utils.is_flash_linear_attention_available()` or `is_causal_conv1d_available()` is false, or if a runtime inspection shows `Qwen3_5MoeGatedDeltaNet` using `torch_causal_conv1d_update` or `torch_chunk_gated_delta_rule`.

`flash-attn` is also required when full-attention layers are configured to use `attn_implementation="flash_attention_2"`.

## Create A Shared Initialization

Use the same randomly initialized weights for AutoEP and ZeRO-3 leaf when comparing loss curves. This makes the metric comparison easier to interpret.

```bash
python prepare_init_weights.py \
  --model qwen3_5 \
  --num_layers 8 \
  --seed 42 \
  --output runs/qwen35/init/qwen35_l8_seed42.safetensors
```

## Run AutoEP

```bash
deepspeed --num_gpus 8 --master_port 29104 train.py \
  --mode autoep \
  --model qwen3_5 \
  --autoep_size 8 \
  --num_layers 8 \
  --steps 100 \
  --warmup_steps 50 \
  --log_interval 1 \
  --seq_len 1024 \
  --micro_batch_size 1 \
  --grad_accum 4 \
  --seed 42 \
  --dataset_name wikitext \
  --dataset_percentage 10.0 \
  --tokenizer_name Qwen/Qwen3-0.6B \
  --load_init_weights runs/qwen35/init/qwen35_l8_seed42.safetensors \
  --metrics_out runs/qwen35/autoep/metrics.csv
```

## Run ZeRO-3 Leaf Baseline

```bash
deepspeed --num_gpus 8 --master_port 29105 train.py \
  --mode zero3_leaf \
  --model qwen3_5 \
  --num_layers 8 \
  --steps 100 \
  --warmup_steps 50 \
  --log_interval 1 \
  --seq_len 1024 \
  --micro_batch_size 1 \
  --grad_accum 4 \
  --seed 42 \
  --dataset_name wikitext \
  --dataset_percentage 10.0 \
  --tokenizer_name Qwen/Qwen3-0.6B \
  --load_init_weights runs/qwen35/init/qwen35_l8_seed42.safetensors \
  --metrics_out runs/qwen35/zero3_leaf/metrics.csv
```

## Compare Metrics

`compare_metrics.py` compares the loss, throughput, and peak memory reported by the two metrics CSV files, then generates summary JSON plus plots.

```bash
python compare_metrics.py \
  --autoep_csv runs/qwen35/autoep/metrics.csv \
  --zero3_leaf_csv runs/qwen35/zero3_leaf/metrics.csv \
  --warmup_steps 50 \
  --out_dir runs/qwen35/compare \
  --out_json runs/qwen35/compare/summary.json
```

Useful outputs:

- `runs/qwen35/compare/summary.json`
- `runs/qwen35/compare/loss_curve.png`
- `runs/qwen35/compare/peak_memory_bar.png`
- `runs/qwen35/compare/throughput_bar.png`

Small numeric differences are expected because AutoEP and ZeRO-3 leaf use different distributed execution paths. The comparison is intended to confirm that loss behavior remains aligned while reporting throughput and memory differences under the same model, data, seed, and initialization.

## Reference Loss Curves

The repository includes reference Qwen3.5 loss-curve images from a longer AutoEP and ZeRO-3 leaf comparison. They are useful for checking the expected curve shape after reproducing the workflow above, but the commands above are the source of truth for a fresh run in your own environment.

![Qwen3.5 CE loss curve](qwen35_aux_10k_ce_loss_curve.png)

![Qwen3.5 total loss curve](qwen35_aux_10k_total_loss_curve.png)

![Qwen3.5 aux loss curve](qwen35_aux_10k_aux_loss_curve.png)
