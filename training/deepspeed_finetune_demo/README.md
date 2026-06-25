# DeepSpeed finetune examples
This finetune example is extracted and modified from [ZenFlow Llama-2 Fine-Tuning Example](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/training/DeepSpeed-ZenFlow/finetuning) in [DeepSpeedExamples](https://github.com/deepspeedai/DeepSpeedExamples).  The purpose is to demostrate how to use different DeepSpeed training features and compare their performance in a single place.

Currently in DeepSpeedExamples, each technology has a dedicated directory to show how to use it.  However, DeepSpeed's philosophy is to allow users to use different features with different configuration file with no code change needed.  This project put this claim to the test.

# How to use

To run the example, simply run:
```
./finetune.sh <NUM_GPUS> <MODEL_NAME> <DS_CONFIG>
```

For example, if we want to run Qwen2.5-3B model with ZeRO offload on 2 GPUs, we can run:
```
./finetune.sh 2 Qwen2.5-3B configs/zo_config.json
```

## Key arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--batch_size` | Training batch size per GPU | required |
| `--eval_batch_size` | Eval batch size per rank | 4 |
| `--eval_steps` | Run evaluation every N steps (0 disables) | 0 |
| `--max_steps` | Stop after N steps (-1 = full epoch) | -1 |
| `--checkpoint_steps` | Save a checkpoint every N steps (0 disables); keeps last 2 | 0 |
| `--wandb_name` | Wandb run name (optional) | None |
| `--num_train_epochs` | Number of training epochs | 3 |
| `--weight_decay` | Weight decay | 0.01 |
| `--warmup` | Warmup ratio | 0.01 |

Note: Learning rate is controlled entirely by the DeepSpeed config JSON, not by command-line arguments.

## Batch size
In DeepSpeed, batch size is decided by configuration file.  However, to avoid modify the config file, this python script takes `--batch_size` parameter and use it to decide train batch size.  Keep this in mind if you need to try different batch size.

## Wandb support
An optional `--wandb_name` can be supplied to finetune_llama.py to generate wandb graph.  But you need to modify `finetune.sh` manually to supply this argument.

## Dataset support

The training script uses a `DATASET_REGISTRY` to configure datasets. Registered datasets are loaded with proper field mapping and preprocessing automatically.

| Dataset | Format | Use Case | Notes |
|---------|--------|----------|-------|
| `sahil2801/CodeAlpaca-20k` | Alpaca | Code instruction tuning | |
| `meta-math/MetaMathQA` | Alpaca | Math reasoning | `sample_rate=0.1` (39.5k of 395k) |
| `cais/mmlu` | MMLU MCQ | Knowledge tasks | Uses `auxiliary_train` split (~95k) |
| `tatsu-lab/alpaca` | Alpaca | General instruction tuning | Fallback default |
| `ise-uiuc/Magicoder-OSS-Instruct-75K` | Magicoder | Code instruction tuning | Auto-detected via `problem` column |

**Registered datasets** are specified by `--dataset_name` directly. **Unregistered datasets** are auto-detected: if the dataset has a `problem` column, Magicoder format is used; otherwise Alpaca format is assumed.

All formats use instruction-masked loss (only the response part contributes to loss).

### Adding a new dataset

Add an entry to `DATASET_REGISTRY` in `finetune_llama.py`:

```python
"your-dataset/name": {
    "split": "train",
    "preprocessor": "alpaca",       # or a custom preprocessor name
    "field_map": {                   # maps source fields to Alpaca format
        "instruction": "source_inst_field",
        "input": None,               # set to None if not present
        "output": "source_output_field",
    },
    "sample_rate": 0.1,             # optional: downsample large datasets
},
```

# Moonlight-16B-A3B with AutoEP + Muon

This project supports fine-tuning [Moonlight-16B-A3B](https://huggingface.co/moonshotai/Moonlight-16B-A3B) (a 16B-parameter MoE model with 3B active parameters) using DeepSpeed AutoEP (automatic expert parallelism) and the Muon optimizer.

## Quick start (8x A100 40GB)

```bash
# 1. Train
deepspeed --num_gpus=8 finetune_llama.py \
  --model_name moonshotai/Moonlight-16B-A3B \
  --output_dir output_moonlight_muon \
  --batch_size 16 --max_length 512 \
  --deepspeed_config configs/z2_moonlight_autoep_muon.json \
  --dataset_name sahil2801/CodeAlpaca-20k \
  --num_train_epochs 1

# 2. Convert DeepSpeed checkpoint to HuggingFace format
python convert_ds_to_hf.py \
  --ds_checkpoint output_moonlight_muon/step_<LAST_STEP> \
  --original_model moonshotai/Moonlight-16B-A3B \
  --output_dir hf_model_muon \
  --ep_size 8

# 3. Generate HumanEval completions
python evaluate/humaneval/gen_humaneval.py \
  --model hf_model_muon \
  --output evalplus_results/muon \
  --instruction

# 4. Evaluate
python -m evalplus.evaluate \
  --dataset humaneval \
  --samples evalplus_results/muon/samples.jsonl
```

## Checkpoint format

With AutoEP, each rank holds a different expert shard. The training script saves checkpoints to `<output_dir>/step_<N>/`:
- `0/model_weights.pt`: full state dict (non-expert params + local experts for rank 0)
- `1/model_weights.pt` ... `7/model_weights.pt`: expert shard params only

Use `convert_ds_to_hf.py` to merge all shards back into a standard HuggingFace model.

## HumanEval results

| Model | HumanEval (base) | HumanEval+ |
|-------|-----------------|------------|
| Moonlight-16B-A3B (baseline) | 46.3% | 40.2% |
| + Muon fine-tune on CodeAlpaca-20k (1 epoch) | 54.9% | 47.0% |

## AutoEP config

AutoEP config goes inside the DeepSpeed JSON under `expert_parallel`:

```json
{
    "expert_parallel": {
        "enabled": true,
        "autoep_size": 8,
        "expert_w1": "gate_proj",
        "expert_w2": "down_proj",
        "expert_w3": "up_proj",
        "route_scale": 2.446,
        "load_balance_coeff": null
    }
}
```

| Parameter | Description |
|-----------|-------------|
| `autoep_size` | Number of expert-parallel ranks (typically = num_gpus) |
| `expert_w1/w2/w3` | Names of the expert weight projections in the HF model |
| `route_scale` | Router output scaling factor (should match `routed_scaling_factor` in model config) |
| `load_balance_coeff` | Auxiliary load-balancing loss coefficient (`null` to disable) |

# Benchmarking

To run benchmark, run:
```
./benchmark.sh <NUM_GPUS> <MODEL_NAME> <DS_CONFIG>
```

# Profiling

To run profiling, run:
```
./profile.sh <NUM_GPUS> <MODEL_NAME> <DS_CONFIG>
```

# Config files

For quick start, some config files are added, you may also modify the config to fit your need.

| Config File | Description |
|-------------|-------------|
| z2_config.json | ZeRO Stage 2 with AdamW |
| z3_config.json | ZeRO Stage 3 with AdamW |
| zo_config.json | ZeRO Offload, stage 2 |
| z3o_config.json | ZeRO Offload, stage 3 |
| zf_config.json | ZeRO Offload with ZenFlow |
| so_config.json | ZeRO Offload with SuperOffload |
| z2_muon.json | ZeRO 2 with Muon optimizer |
| z3_muon.json | ZeRO 3 with Muon optimizer |
| tp_config.json | ZeRO 2 with AutoTP |
| z2_moonlight_autoep_adam.json | Moonlight-16B-A3B with AutoEP + AdamW |
| z2_moonlight_autoep_muon.json | Moonlight-16B-A3B with AutoEP + Muon |

## Muon optimizer config

Muon is a hybrid optimizer: it applies Muon updates to 2D hidden weights and Adam to everything else.  The config supports separate learning rates:

```json
{
    "optimizer": {
        "type": "Muon",
        "params": {
            "muon_lr": 1e-3,
            "adam_lr": 2e-5,
            "momentum": 0.95,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    }
}
```

| Parameter | Description |
|-----------|-------------|
| `muon_lr` | Learning rate for Muon (2D hidden weights) |
| `adam_lr` | Learning rate for Adam (embeddings, layer norms, lm_head, etc.) |
| `momentum` | Muon momentum factor |
| `betas` | Adam betas (for non-Muon parameters) |
| `eps` | Adam epsilon |
| `weight_decay` | Weight decay for both Muon and Adam parameters |
