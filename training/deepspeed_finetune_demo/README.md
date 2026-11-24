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

## Validate AutoEP checkpoint resume

`run_autoep_affine_ir_checkpoint_experiment.sh` compares the loss from uninterrupted
training with loss after resuming from both a native DeepSpeed checkpoint and a
Universal checkpoint. It runs each path through step 100, saves a native
checkpoint at step 50, converts that checkpoint to Universal format, and compares
the losses for steps 51-100.

Run the experiment on a Linux host with the required GPUs and dependencies
installed. Set `DEEPSPEED_REPO` to the root of a DeepSpeed checkout that provides
`deepspeed/checkpoint/ds_to_universal.py`. The model and dataset must already be
cached because Transformers and Datasets offline mode are enabled by default:

```bash
cd training/deepspeed_finetune_demo
export DEEPSPEED_REPO=/path/to/DeepSpeed
./run_autoep_affine_ir_checkpoint_experiment.sh
```

The script defaults to 8 GPUs, AutoEP size 8, ZeRO stage 2, the
`moonshotai/Moonlight-16B-A3B` model, and the `tatsu-lab/alpaca` dataset. Override
these settings with environment variables as needed:

```bash
NUM_GPUS=8 AUTOEP_SIZE=8 ZERO_STAGE=2 \
MODEL_NAME=moonshotai/Moonlight-16B-A3B \
DATASET_NAME=tatsu-lab/alpaca \
OUTPUT_ROOT=/path/to/experiment \
./run_autoep_affine_ir_checkpoint_experiment.sh
```

`DEEPSPEED_LAUNCHER` defaults to `ds`; set it to `deepspeed` if that is the
launcher available in your environment. To use GPUs selected by DeepSpeed's
`--include` option, set `GPU_INCLUDE` (for example, `GPU_INCLUDE=localhost:0,1`).
To allow downloads instead of using cached model and dataset files, set
`TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0`.

The output directory contains the run logs, generated DeepSpeed configs,
`loss_comparison.csv` with per-step losses, and `loss_comparison.txt` with the
maximum and mean absolute loss differences from the uninterrupted baseline. A
successful run means both resume paths produced comparable losses; inspect the
logs and differences when diagnosing a mismatch.

## AutoEP from HF checkpoint to rollout

`autoep_rollout_e2e.py` follows weights from a Hugging Face checkpoint through AutoEP
training to a rollout engine with a different expert layout, and checks precision at
every step:

1. Load the checkpoint, truncated to `NUM_LAYERS` (4 by default), into an AutoEP
   training engine (`TRAIN_AUTOEP_SIZE`, default 4). Every weight must equal the
   safetensors tensor bit for bit; logits are compared with plain transformers.
2. Train a few steps.
3. Move the trained weights to rollout engines (`ROLLOUT_AUTOEP_SIZE`, default 8) two ways:
   - `ucp`: `save_checkpoint`, `ds_to_universal`, then `load_checkpoint` with `load_universal`;
   - `affine`: AutoEP storage maps, `plan_transfer` and `execute_transfers` for the
     experts, directly between devices; dense weights are copied on device (gathered
     under ZeRO-3).
4. Generate greedily with `HybridEngineRollout` on the training engine and on both
   rollout engines.

### Weight flows

The two paths of step 3 next to the existing in-place rollout:

```
Universal checkpoint (offline, through storage)
  training engine          save_checkpoint     ZeRO shards     ds_to_universal     universal      load_checkpoint     rollout engine
  EP=4, ZeRO-k     ───────────────────────────>  on disk   ──────────────────────>  checkpoint  ──(load_universal)──>  EP=8, NaN init
                                                                                    on disk

AffineIR transfer (online, device to device)
  training engine ── expert_storage_affine_maps() ──> source maps ─┐
  (live storage: EDP replicas, ZeRO-3 fragments)                   ├─ plan_transfer ─> TransferPlan ─ execute_transfers ─> rollout experts
  rollout engine  ── expert_storage_affine_maps() ──> target maps ─┘   (per tensor)                  (send/recv + local copy)
  dense weights: copy_ on device (GatheredParameters under ZeRO-3)

HybridEngineRollout (in place, reference)
  training engine = rollout engine; sync_weights is a no-op, so the layout cannot change
```

|                        | Universal checkpoint        | AffineIR transfer                   | HybridEngineRollout     |
|------------------------|-----------------------------|-------------------------------------|-------------------------|
| Target layout          | any EP size                 | any EP size or expert placement     | same as training        |
| Goes through storage   | yes, full checkpoint twice  | no                                  | no                      |
| Source                 | saved ZeRO shards           | live parameter storage              | live parameters         |
| Moves                  | all weights and, optionally, optimizer state | expert weights (dense copied separately) | nothing        |

Measured on 8x Intel Arc Pro B70, Qwen3-30B-A3B truncated to 4 layers, training at EP4,
rollout at EP8 (UCP is not run for ZeRO-3, see below):

| Step                               | UCP, ZeRO-2  | AffineIR, ZeRO-2 | AffineIR, ZeRO-3 |
|------------------------------------|--------------|------------------|------------------|
| save / plan                        | 7.1 s        | 3.2 s            | 0.05 s           |
| convert / execute                  | 107.7 s      | 0.27 s           | 0.23 s           |
| load / dense copy                  | 25.1 s       | 0.01 s           | 0.18 s           |
| rollout weights equal trained      | bitwise      | bitwise          | bitwise          |

Each check covers 8.0e9 elements. AffineIR sent 4.2 GB between devices and copied 0.6 GB
locally under ZeRO-2 (3.6 GB and 1.2 GB under ZeRO-3), in 12 rounds. Under ZeRO-3 the
dense copy gathers each weight.

### Checks

Rollout engines start from NaN weights. The script exits with an error unless:

- every training weight equals the safetensors tensor bit for bit after loading;
- every rollout weight equals the trained weight bit for bit (experts are matched by
  expert id through plain collectives);
- the two paths' logits differ by no more than the forward's own run-to-run difference,
  which is zero when the forward is deterministic.

Logits and tokens against the training engine are reported but not required to match
bit for bit. The expert layout changes how tokens are grouped in the grouped GEMM, and
with it the bf16 accumulation order, so the training and rollout forwards round
differently even with identical weights. Set `ROLLOUT_AUTOEP_SIZE` equal to
`TRAIN_AUTOEP_SIZE` to remove this difference.

Each stage writes `zero<stage>.json` with the checks, the time of every step (save,
convert, load; plan, execute, dense copy; generation) and the generated text.

```bash
cd training/deepspeed_finetune_demo
MODEL_NAME=/path/to/Qwen3-30B-A3B ./run_autoep_rollout_e2e.sh
```

`NUM_GPUS`, `NUM_LAYERS`, `TRAIN_AUTOEP_SIZE`, `ROLLOUT_AUTOEP_SIZE`, `ZERO_STAGES`
(default `"2 3"`), and `OUTPUT_ROOT` override the defaults. The script needs a DeepSpeed
build with the AutoEP storage maps
([DeepSpeed#8800](https://github.com/deepspeedai/DeepSpeed/pull/8800)) and the AffineIR
transfer executor ([DeepSpeed#8809](https://github.com/deepspeedai/DeepSpeed/pull/8809)).
ZeRO-3 training with AutoEP also needs the fix that makes ZeRO-3 gather expert and dense
parameters in the same order on every rank
([DeepSpeed#8876](https://github.com/deepspeedai/DeepSpeed/pull/8876)); without it, training can hang from the third
step. A universal checkpoint converted from ZeRO-3 keeps ZeRO-3 model file names, which
the ZeRO-1 rollout engine cannot load, so the run script uses only the AffineIR path for
ZeRO-3. The UCP path writes a full training checkpoint and its universal form under
`OUTPUT_ROOT` (about 90 GB per stage for 4 layers of Qwen3-30B-A3B), removed after each
stage.

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
