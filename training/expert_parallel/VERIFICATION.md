# Qwen3.5 AutoEP Verification

This procedure verifies the current expert-parallel sample with the `qwen3_5`
model preset, Hugging Face Wikitext data, an 8-layer Qwen3.5-MoE-style model,
shared initialization weights, and an AutoEP vs HF ZeRO-3 leaf comparison.

The verification run for dev_ds issue 78 used:

- artifact root: `/mnt/local_storage/autoep_current_sample_qwen35_20260501`
- sample path: `/home/ray/default/dev_ds/DeepSpeedExamples/training/expert_parallel`
- DeepSpeed path on `PYTHONPATH`: `/mnt/user_storage/ds_gittrees/verify-autoep-qwen35-release-evidence`
- model preset: `qwen3_5`
- AutoEP preset: `qwen3_5_moe`
- layers: `8`
- dataset: `wikitext`, `dataset_percentage=10.0`
- tokenizer: `Qwen/Qwen3-0.6B`
- sequence length: `128`
- micro batch size: `2`
- gradient accumulation: `1`
- world size: `8`
- effective tokens per update: `2048`
- seed: `42`
- long run steps: `10000`, warmup steps: `50`
- shared init SHA256: `97760d09b132f697de7f6ccf6d2d70070fe1def048b8495b0870c2c49a40bf81`

AutoEP uses the sample's built-in config for `--mode autoep`: bf16, AdamW,
ZeRO stage 1, `expert_parallel.enabled=true`, `autoep_size=8`, and
`preset_model=qwen3_5_moe`. The baseline uses `--mode zero3_leaf`: bf16, AdamW,
ZeRO stage 3, and a ZeRO leaf module for
`transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeSparseMoeBlock`.

The Wikitext loader filters empty/whitespace-only text rows before tokenization.
Without that filter, Wikitext rows containing only padding can produce all
`-100` labels and a non-finite first loss.

## Environment

Run from:

```bash
cd /home/ray/default/dev_ds/DeepSpeedExamples/training/expert_parallel
export PYTHONPATH=/mnt/user_storage/ds_gittrees/verify-autoep-qwen35-release-evidence${PYTHONPATH:+:$PYTHONPATH}
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/local_storage/hf_cache
export HF_DATASETS_CACHE=/mnt/local_storage/hf_datasets_cache
RUN_ROOT=/mnt/local_storage/autoep_current_sample_qwen35_20260501
SMOKE_ROOT=$RUN_ROOT/smoke_100
LONG_ROOT=$RUN_ROOT/long_10000
INIT=$SMOKE_ROOT/init_weights_qwen35_current_8l_seed42.safetensors
LONG_INIT=$LONG_ROOT/init_weights_qwen35_current_8l_seed42.safetensors
```

## Import Provenance

Record package paths before launching:

```bash
conda run --no-capture-output -n ds python - <<'PY'
import json
import socket
import torch
import transformers
import deepspeed
import train

preset = train.MODEL_PRESETS["qwen3_5"]
payload = {
    "hostname": socket.gethostname(),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "transformers_version": transformers.__version__,
    "transformers_file": transformers.__file__,
    "deepspeed_version": getattr(deepspeed, "__version__", "unknown"),
    "deepspeed_file": deepspeed.__file__,
    "qwen3_5_preset": preset,
    "autoep_config": train.build_default_deepspeed_config("autoep", preset["architecture"]),
    "zero3_leaf_config": train.build_default_deepspeed_config("zero3_leaf", preset["architecture"]),
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
```

## Shared Init

Generate one pre-DeepSpeed initialization artifact and reuse it for both modes:

```bash
mkdir -p "$SMOKE_ROOT/results" "$LONG_ROOT/results"

conda run --no-capture-output -n ds python train.py \
  --model qwen3_5 \
  --mode autoep \
  --num_layers 8 \
  --seq_len 128 \
  --micro_batch_size 2 \
  --grad_accum 1 \
  --seed 42 \
  --init_weights_only \
  --save_init_weights "$INIT"

sha256sum "$INIT" | tee "$SMOKE_ROOT/init_weights.sha256"
ln -f "$INIT" "$LONG_INIT"
ln -f "${INIT%.safetensors}_meta.json" "${LONG_INIT%.safetensors}_meta.json"
```

## 100-Step Smoke

Run the smoke first and only continue to 10,000 steps if both modes and the
strict init-hash comparison pass.

```bash
conda run --no-capture-output -n ds deepspeed --num_gpus 8 --master_port 32701 train.py \
  --model qwen3_5 \
  --mode autoep \
  --num_layers 8 \
  --steps 100 \
  --warmup_steps 10 \
  --log_interval 1 \
  --seq_len 128 \
  --micro_batch_size 2 \
  --grad_accum 1 \
  --seed 42 \
  --dataset_name wikitext \
  --dataset_percentage 10.0 \
  --load_init_weights "$INIT" \
  --metrics_out "$SMOKE_ROOT/metrics_autoep.csv" \
  --run_metadata_out "$SMOKE_ROOT/metadata_autoep.json"

conda run --no-capture-output -n ds deepspeed --num_gpus 8 --master_port 32711 train.py \
  --model qwen3_5 \
  --mode zero3_leaf \
  --num_layers 8 \
  --steps 100 \
  --warmup_steps 10 \
  --log_interval 1 \
  --seq_len 128 \
  --micro_batch_size 2 \
  --grad_accum 1 \
  --seed 42 \
  --dataset_name wikitext \
  --dataset_percentage 10.0 \
  --load_init_weights "$INIT" \
  --metrics_out "$SMOKE_ROOT/metrics_zero3_leaf.csv" \
  --run_metadata_out "$SMOKE_ROOT/metadata_zero3_leaf.json"

conda run --no-capture-output -n ds python compare_metrics.py \
  --autoep_csv "$SMOKE_ROOT/metrics_autoep.csv" \
  --zero3_leaf_csv "$SMOKE_ROOT/metrics_zero3_leaf.csv" \
  --autoep_metadata "$SMOKE_ROOT/metadata_autoep.json" \
  --zero3_leaf_metadata "$SMOKE_ROOT/metadata_zero3_leaf.json" \
  --out_dir "$SMOKE_ROOT/results" \
  --out_json "$SMOKE_ROOT/results/summary.json" \
  --autoep_label "AutoEP + ZeRO-1" \
  --zero3_leaf_label "HF + ZeRO-3 leaf" \
  --warmup_steps 10 \
  --require_same_init_hash
```

## 10,000-Step Verification

Run the full comparison with the same initialization artifact:

```bash
conda run --no-capture-output -n ds deepspeed --num_gpus 8 --master_port 32721 train.py \
  --model qwen3_5 \
  --mode autoep \
  --num_layers 8 \
  --steps 10000 \
  --warmup_steps 50 \
  --log_interval 1 \
  --seq_len 128 \
  --micro_batch_size 2 \
  --grad_accum 1 \
  --seed 42 \
  --dataset_name wikitext \
  --dataset_percentage 10.0 \
  --load_init_weights "$LONG_INIT" \
  --metrics_out "$LONG_ROOT/metrics_autoep.csv" \
  --run_metadata_out "$LONG_ROOT/metadata_autoep.json"

conda run --no-capture-output -n ds deepspeed --num_gpus 8 --master_port 32731 train.py \
  --model qwen3_5 \
  --mode zero3_leaf \
  --num_layers 8 \
  --steps 10000 \
  --warmup_steps 50 \
  --log_interval 1 \
  --seq_len 128 \
  --micro_batch_size 2 \
  --grad_accum 1 \
  --seed 42 \
  --dataset_name wikitext \
  --dataset_percentage 10.0 \
  --load_init_weights "$LONG_INIT" \
  --metrics_out "$LONG_ROOT/metrics_zero3_leaf.csv" \
  --run_metadata_out "$LONG_ROOT/metadata_zero3_leaf.json"

conda run --no-capture-output -n ds python compare_metrics.py \
  --autoep_csv "$LONG_ROOT/metrics_autoep.csv" \
  --zero3_leaf_csv "$LONG_ROOT/metrics_zero3_leaf.csv" \
  --autoep_metadata "$LONG_ROOT/metadata_autoep.json" \
  --zero3_leaf_metadata "$LONG_ROOT/metadata_zero3_leaf.json" \
  --out_dir "$LONG_ROOT/results" \
  --out_json "$LONG_ROOT/results/summary.json" \
  --autoep_label "AutoEP + ZeRO-1" \
  --zero3_leaf_label "HF + ZeRO-3 leaf" \
  --warmup_steps 50 \
  --require_same_init_hash
```

The final review surface for issue 78 is
`todo/docs/verify-autoep-qwen35-release-evidence/results.md` in the dev_ds
repository. The large CSV and log artifacts stay under `/mnt/local_storage/`.
