# AutoEP Training Example

AutoEP (Auto Expert Parallelism) automatically partitions MoE expert weights across GPUs and uses AllToAll communication to route tokens to the correct experts.
This example offers a quick start for AutoEP in DeepSpeed.

## Quick Start

### Prerequisites

- 2+ GPUs (the fast grouped GEMM path works only on Hopper and Blackwell GPUs)
- Dependencies:
  - PyTorch `>= 2.9.1`
  - **DeepSpeed** with AutoEP: AutoEP has not been merged into `main` yet. `requirements.txt` installs the **tip of [PR #7938](https://github.com/deepspeedai/DeepSpeed/pull/7938)**. Manual install: `pip install "git+https://github.com/deepspeedai/DeepSpeed.git@refs/pull/7938/head#egg=deepspeed"`.
  - **`transformers` `>= 5.2`** (see `requirements.txt`)
  - Qwen3.5 requires specific kernel dependencies. See [VERIFICATION.md](VERIFICATION.md#qwen35-kernel-requirements) for more details.
  - See `requirements.txt` for other dependencies.

### Run

The following launches causal LM training with **AutoEP + ZeRO-1** on a randomly initialized model built from the original **Qwen3.5-MoE** Hugging Face text config.
`--num_layers` overrides only the layer count in the original model config, which is useful when testing with limited GPU resources. `--dataset_name` and `--dataset_percentage` choose the Hugging Face training dataset and the percentage of the train split to use.

```bash
deepspeed --num_gpus 8 train.py \
    --mode autoep \
    --model qwen3_5_moe \
    --autoep_size 8 \
    --num_layers 8 \
    --dataset_name wikitext \
    --dataset_percentage 10.0 \
    --steps 1000
```

For this `--mode autoep` / `--model qwen3_5_moe` run, `train.py` derives the following `expert_parallel` section:

```json
    "expert_parallel": {
        "enabled": true,
        "autoep_size": 8,
        "preset_model": "qwen3_5_moe"
    }
```

Here are the key options in the DeepSpeed config for AutoEP:

- **`enabled`** — Turns on AutoEP.
- **`autoep_size`** — Expert-parallel size. It must be specified with `--autoep_size` in AutoEP mode and must divide both the GPU count and the model's expert count. The benchmark commands use `8` for Qwen3.5 and `4` for Llama4 and Mixtral.
- **`preset_model`** — DeepSpeed's structural AutoEP preset id. The example's public `--model` choices intentionally use the same ids when an AutoEP preset exists.

This example exposes three public `--model` choices: `qwen3_5_moe`, `llama4`, and `mixtral`. These match the DeepSpeed `preset_model` ids used by AutoEP for the same structures. The underlying AutoEP PR also defines additional structural preset ids: `qwen3_moe`, `deepseek_v2`, and `deepseek_v3`; those are not exposed as `--model` choices in this example.

## Performance Benchmark

We benchmarked Qwen3.5 and Mixtral with 8 layers, and Llama4 with 7 layers. Each model was run under matching conditions for AutoEP and the ZeRO-3 leaf baseline: 8 H100 GPUs, sequence length 1024, micro batch size 1, gradient accumulation 4, 100 optimizer steps, and steps 50-99 measured. The table below reports the side-by-side comparison. Llama4 uses 7 layers because the 8-layer ZeRO-3 leaf baseline OOMed during backward.

| Model | ZeRO-3 leaf | AutoEP (+ZeRO-1) |
| --- | --- | --- |
| Qwen3.5 MoE | 42,128.05 tok/s, 34.99 GB | 87,540.15 tok/s, 25.58 GB (`2.08x` throughput, `0.73x` memory vs ZeRO-3) |
| Llama4 (7 layers) | 19,144.07 tok/s, 56.95 GB | 60,178.91 tok/s, 60.08 GB (`3.14x` throughput, `1.06x` memory vs 7-layer ZeRO-3) |
| Mixtral 8x7B | 32,622.11 tok/s, 50.47 GB | 69,052.31 tok/s, 35.03 GB (`2.12x` throughput, `0.69x` memory vs ZeRO-3) |

Qwen3.5 reproduction steps and reference loss curves for the AutoEP and ZeRO-3 leaf comparison are in [VERIFICATION.md](VERIFICATION.md).


## Important Constraints

### `autoep_size` requirements

- Must be `<= num_experts`
- Must evenly divide `num_experts`
- Must evenly divide `world_size`
- `autoep_size=1` bypasses EP communication entirely (degenerate case)

### Grouped GEMM backend

`torch._grouped_mm` is required for the default production path. With the default `expert_parallel.use_grouped_mm=true`, DeepSpeed fails fast if `torch._grouped_mm` is unavailable. Set `use_grouped_mm=false` in the DeepSpeed config only for functional/debug runs that intentionally use the sequential for-loop path. On A100 (SM80), verify availability and actual throughput since the Hopper fast path may not activate.

### Qwen3.5 linear-attention kernels

For `--model qwen3_5_moe`, the required linear-attention kernel dependencies and verification checks are documented in [VERIFICATION.md](VERIFICATION.md#qwen35-kernel-requirements).

### bf16 requirement

`bf16` is recommended. `fp16` is functionally correct but not optimized for the Hopper grouped-GEMM fast path used by `torch._grouped_mm`.

### Optimizer wiring

AutoEP runs must let DeepSpeed build the optimizer from the JSON config (no client optimizer). This ensures `configure_moe_param_groups()` is invoked to split expert parameters into expert-data-parallel reduction groups.

### Load balancing status

DeepSeek-style auxiliary-loss-free (expert-bias) load balancing is **not yet implemented** in AutoEP.
