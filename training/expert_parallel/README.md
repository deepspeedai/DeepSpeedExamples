# AutoEP Training Example

AutoEP (Auto Expert Parallelism) automatically partitions MoE expert weights across GPUs and uses AllToAll communication to route tokens to the correct experts.
This example offers a quick start for AutoEP in DeepSpeed.

## Quick Start

### Prerequisites

- 2+ GPUs (The fast grouped GEMM path works only on Hopper and Blackwell GPUs)
- Dependencies:
  - PyTorch `>= 2.9.1`
  - **DeepSpeed** with AutoEP: `requirements.txt` installs the **tip of [PR #7938](https://github.com/deepspeedai/DeepSpeed/pull/7938)**.
 Manual install: `pip install "git+https://github.com/deepspeedai/DeepSpeed.git@refs/pull/7938/head#egg=deepspeed"`.
  - **`transformers` `>= 5.6.2`** (5.x line; see `requirements.txt`) — stable patch floor that includes Mixtral, Llama 4, Qwen3 MoE, and Qwen3.5 / Qwen3.6 (`qwen3_5_moe`) in current Hub checkpoints. Newer 5.7+ releases will be fine.
  - **Qwen3.5 kernel dependencies are mandatory for verification**:
    `flash-linear-attention` (`import fla`) and `causal-conv1d` (`import causal_conv1d`) must be installed so linear-attention layers use the specialized `fla.ops.gated_delta_rule` and `causal_conv1d` kernels, not the Transformers torch fallbacks. `flash-attn` (`import flash_attn`) must also be importable for Qwen3.5 full-attention layers when the FlashAttention2 attention implementation is requested.
  - See `requirements.txt` for other dependencies.

### Run

The following launches causal LM training with **AutoEP + ZeRO-1** on a randomly initialized **Qwen3.5-MoE**-style text model and **Hugging Face text** data (default corpus: **WikiText-103**).

```bash
deepspeed --num_gpus 8 train.py \
    --model qwen3_5 \
    --dataset_name wikitext \
    --dataset_percentage 10.0 \
    --steps 1000
```

`train.py` builds the DeepSpeed config from **`--mode`** (default `autoep`) and **`--model`**, so **`expert_parallel.preset_model`** and ZeRO-3 **`leaf_module.classes`** always match the Hugging Face MoE layout for that preset. For **Qwen3.5-MoE** (`--model qwen3_5`), the effective AutoEP section is equivalent to:

```json
    "expert_parallel": {
        "enabled": true,
        "autoep_size": 8,
        "preset_model": "qwen3_5_moe",
        "load_balance_coeff": null
    }
```

Here **`preset_model` is `qwen3_5_moe`**: the **AutoEP structural preset id** in DeepSpeed (how layers are found and experts are wired). Mixtral-style presets use **`preset_model`: `mixtral`** and default **`autoep_size`**: `4`.

Optional **`--deepspeed_config path.json`** loads a JSON file and then overwrites only the architecture-specific fields above (and the ZeRO-3 leaf MoE block class) so a stray preset in the file cannot disagree with **`--model`**.

Files under `configs/` remain as **reference overrides** (optimizer, scheduler, batch defaults).


Batches come from a Hugging Face **text** dataset. **`--dataset_name` defaults to `wikitext`** (WikiText-103 raw) and **`--dataset_percentage` defaults to `10.0`** (ten percent of the train split), matching the presets in [`ds_verify_loss`](https://github.com/tohtana/ds_verify_loss). Override **`--dataset_name`**, **`--dataset_percentage`**, or **`--tokenizer_name`** as needed (the tokenizer vocabulary must match the model config; the `qwen3_5` preset defaults to the **Qwen3-0.6B** tokenizer, `len(tokenizer)=151669`).

## Observed Results

### Loss curve

![Loss Curve Comparison](loss_curve.png)

### Peak memory

![Peak Memory Comparison](peak_memory_bar.png)

### Throughput

![Throughput Comparison](throughput_bar.png)

| Metric | AutoEP + ZeRO-1 | HF + ZeRO-3 leaf |
|--------|------------------|-------------------|
| Final loss (step 999) | 10.415 | 10.418 |
| Peak GPU memory | 55.32 GB | 75.36 GB |
| Avg throughput (post-warmup) | 8,997 tok/s | 2,189 tok/s |

| Comparison metric | Value |
|-------------------|-------|
| Final abs loss diff | 0.0031 |
| Last 100-step mean abs loss diff | 0.0085 |
| Mean abs loss diff (950 aligned steps) | 0.0222 |
| Max abs loss diff | 0.1428 |
| Pearson correlation | 0.9958 |
| Memory ratio (autoep/zero3) | 0.73x |
| Throughput ratio (autoep/zero3) | 4.11x |
| Same init hash | true |


## Important Constraints

### `autoep_size` requirements

- Must be `<= num_experts`
- Must evenly divide `num_experts`
- Must evenly divide `world_size`
- `autoep_size=1` bypasses EP communication entirely (degenerate case)

### Grouped GEMM backend

`torch._grouped_mm` is required for production performance. Without it, the code falls back to a sequential for-loop over experts. On A100 (SM80), verify availability and actual throughput since the Hopper fast path may not activate.

### Qwen3.5 linear-attention kernels

For `--model qwen3_5`, `flash-linear-attention` and `causal-conv1d` are verification requirements, not optional accelerators. The verification should fail if `transformers.utils.import_utils.is_flash_linear_attention_available()` or `is_causal_conv1d_available()` is false, or if a runtime inspection shows `Qwen3_5MoeGatedDeltaNet` using `torch_causal_conv1d_update` or `torch_chunk_gated_delta_rule`.

`flash-attn` is also required when full-attention layers are configured to use `attn_implementation="flash_attention_2"`; record the active attention implementation in verification metadata.


### bf16 requirement

`bf16` is recommended. `fp16` is functionally correct but not optimized for the Hopper grouped-GEMM fast path used by `torch._grouped_mm`.

### Optimizer wiring

AutoEP runs must let DeepSpeed build the optimizer from the JSON config (no client optimizer). This ensures `configure_moe_param_groups()` is invoked to split expert parameters into expert-data-parallel reduction groups.

### Load balancing status

`load_balance_coeff` is accepted in config but the bias update pre-hook is **not yet implemented**. Setting it has no runtime effect (`expert_bias` stays at zero). The `AutoEPConfig` default is `1e-3`, so explicitly set `null` to avoid registering an unused buffer.
