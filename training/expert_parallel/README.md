# AutoEP Training Example

AutoEP (Auto Expert Parallelism) automatically partitions MoE expert weights across GPUs and uses AllToAll communication to route tokens to the correct experts.
This example offers a quick start for AutoEP in DeepSpeed.

## Quick Start

### Prerequisites

- 2+ GPUs (the fast grouped GEMM path works only on Hopper and Blackwell GPUs; the Qwen3.5 fast-path verification target is H100 with Triton `>= 3.4`)
- Dependencies:
  - PyTorch `>= 2.9.1`
  - **DeepSpeed** with AutoEP: `requirements.txt` installs the **tip of [PR #7938](https://github.com/deepspeedai/DeepSpeed/pull/7938)**.
 Manual install: `pip install "git+https://github.com/deepspeedai/DeepSpeed.git@refs/pull/7938/head#egg=deepspeed"`.
  - **`transformers` `>= 5.6.2`** (5.x line; see `requirements.txt`) — stable patch floor that includes Mixtral, Llama 4, Qwen3 MoE, and Qwen3.5 / Qwen3.6 (`qwen3_5_moe`) in current Hub checkpoints. Newer 5.7+ releases will be fine.
  - **Qwen3.5 requires specific kernel dependencies**:
    `flash-linear-attention` (`import fla`) and `causal-conv1d` (`import causal_conv1d`) must be installed so linear-attention layers use the specialized `fla.ops.gated_delta_rule` and `causal_conv1d` kernels, not the Transformers torch fallbacks. `flash-attn` (`import flash_attn`) must also be importable for Qwen3.5 full-attention layers when the FlashAttention2 attention implementation is requested. On H100 with Triton `>= 3.4`, `tilelang` (`import tilelang`) is also required for the Qwen3.5 fast path.
  - See `requirements.txt` for other dependencies.

### Run

The following launches causal LM training with **AutoEP + ZeRO-1** on a randomly initialized model built from the original **Qwen3.5-MoE** Hugging Face text config and **Hugging Face text** data (default corpus: **WikiText-103**).

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

Here **`preset_model` is `qwen3_5_moe`**: the **AutoEP structural preset id** in DeepSpeed (how layers are found and experts are wired), not a model-size preset. Public model presets are original Hugging Face model families: `qwen3_5`, `llama4`, and `mixtral_8x7b`. They use the original config dimensions by default; **only `--num_layers` overrides depth**, leaving hidden size, expert count, vocabulary size, and other dimensions unchanged. Llama4 uses **`preset_model`: `llama4`** and Mixtral uses **`preset_model`: `mixtral`**; both default to **`autoep_size`: `4`.

Optional **`--deepspeed_config path.json`** loads a JSON file and then overwrites only the architecture-specific fields above (and the ZeRO-3 leaf MoE block class) so a stray preset in the file cannot disagree with **`--model`**.

Files under `configs/` remain as **reference overrides** (optimizer, scheduler, batch defaults).


Batches come from a Hugging Face **text** dataset. **`--dataset_name` defaults to `wikitext`** (WikiText-103 raw) and **`--dataset_percentage` defaults to `10.0`** (ten percent of the train split), matching the presets in [`ds_verify_loss`](https://github.com/tohtana/ds_verify_loss). Override **`--dataset_name`**, **`--dataset_percentage`**, or **`--tokenizer_name`** as needed (the tokenizer length must fit within the model config vocabulary; the `qwen3_5` preset defaults to the **Qwen3-0.6B** tokenizer, `len(tokenizer)=151669`).

## Observed Results

Observed-results benchmarks were run on May 2, 2026 with 8 GPUs, 8 layers, sequence length 1024, micro batch size 1, gradient accumulation 4, `--output_router_logits true`, 100 optimizer steps, and steps 50-99 measured. Matrix artifacts are under `/mnt/local_storage/qwen35_kernel_speed_aux_20260502/task_e_observed_results_models`; the fixed Mixtral AutoEP rerun is under `/mnt/local_storage/qwen35_kernel_speed_aux_20260502/task_f_mixtral_autoep_gate_logits_fix`. Reproducible Qwen3.5 environment details and 10k aux-loss charts are in [VERIFICATION.md](VERIFICATION.md).

| Model | ZeRO-3 leaf | AutoEP (+ZeRO-1) |
| --- | --- | --- |
| Qwen3.5 MoE | Complete: 42,128.05 tok/s, 34.99 GB | Complete: 87,540.15 tok/s, 25.58 GB (`2.08x` throughput, `0.73x` memory vs ZeRO-3) |
| Llama4 | Failed before metrics: CUDA OOM during backward on rank 4 while 78.12 GiB was in use and a 2.50 GiB allocation was requested. Log: `/mnt/local_storage/qwen35_kernel_speed_aux_20260502/task_e_observed_results_models/runs/llama4/zero3_leaf/train.log`. | Complete: 53,927.17 tok/s, 66.68 GB. ZeRO-3 failed, so no ratio is reported. |
| Mixtral 8x7B | Complete: 32,622.11 tok/s, 50.47 GB | Complete: 69,052.31 tok/s, 35.03 GB (`2.12x` throughput, `0.69x` memory vs ZeRO-3) |


## Important Constraints

### `autoep_size` requirements

- Must be `<= num_experts`
- Must evenly divide `num_experts`
- Must evenly divide `world_size`
- `autoep_size=1` bypasses EP communication entirely (degenerate case)

### Grouped GEMM backend

`torch._grouped_mm` is required for production performance. Without it, the code falls back to a sequential for-loop over experts. On A100 (SM80), verify availability and actual throughput since the Hopper fast path may not activate.

### Qwen3.5 linear-attention kernels

For `--model qwen3_5`, `flash-linear-attention`, `causal-conv1d`, `flash-attn`, and `tilelang` on H100/Triton `>= 3.4` are verification requirements, not optional accelerators. The verification should fail if `transformers.utils.import_utils.is_flash_linear_attention_available()` or `is_causal_conv1d_available()` is false, or if a runtime inspection shows `Qwen3_5MoeGatedDeltaNet` using `torch_causal_conv1d_update` or `torch_chunk_gated_delta_rule`.

`flash-attn` is also required when full-attention layers are configured to use `attn_implementation="flash_attention_2"`; record the active attention implementation in verification metadata.


### bf16 requirement

`bf16` is recommended. `fp16` is functionally correct but not optimized for the Hopper grouped-GEMM fast path used by `torch._grouped_mm`.

### Optimizer wiring

AutoEP runs must let DeepSpeed build the optimizer from the JSON config (no client optimizer). This ensures `configure_moe_param_groups()` is invoked to split expert parameters into expert-data-parallel reduction groups.

### Load balancing status

`load_balance_coeff` is accepted in config but the bias update pre-hook is **not yet implemented**. Setting it has no runtime effect (`expert_bias` stays at zero). The `AutoEPConfig` default is `1e-3`, so explicitly set `null` to avoid registering an unused buffer.
