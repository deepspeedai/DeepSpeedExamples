# Reflow Fine-Tuning Examples

Fine-tune large language models with [DeepSpeed](https://www.deepspeed.ai/) ZeRO Stage 3 + **Reflow**, an asynchronous CPU-offload optimizer for mixed-precision FP16/BF16 training. Reflow keeps the optimizer state and FP32 master weights on the CPU like ZeRO-Offload, but overlaps the CPU optimizer work with the backward pass instead of running it serially afterward. It uses the **same GPU memory** as ZeRO-Offload and **~12% less host RAM** (OPT-30B: ~549 vs ~625 GB) — gradients are offloaded in half precision (FP16/BF16) and promoted to FP32 inside the CPU kernel, so there is no CPU-side FP32 gradient buffer (see [Memory](#memory) for measured numbers).

Reflow supports **Adam/AdamW and Lion** with FP16/BF16 model parameters and gradients, FP32 master weights, and FP32 optimizer states. The same scripts run the Reflow path or the plain ZeRO-Offload baseline. `check_bitexact.sh` compares deterministic per-step losses for a selected model and configuration.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# The runtime change is currently proposed in deepspeedai/DeepSpeed#8719.
git clone https://github.com/deepspeedai/DeepSpeed.git
git -C DeepSpeed fetch origin pull/8719/head:reflow
git -C DeepSpeed switch reflow
pip install -e ./DeepSpeed
```

Reflow is a DeepSpeed runtime feature, so you also need a DeepSpeed build that includes it — the `deepspeed.runtime.reflow` module and the `reflow_*` CPU-Adam/Lion ops (JIT-built on first use). No custom modeling code is required: the examples fine-tune plain Hugging Face Transformers models (`--model_name`) on `tatsu-lab/alpaca` by default (override with `--dataset_name`).


The launch scripts default to FlashAttention 2. Install it separately after PyTorch:

```bash
pip install flash-attn --no-build-isolation
```

Alternatively, set `ATTN_IMPLEMENTATION=sdpa` or `eager` when running a launch script; these use PyTorch attention and do not require `flash-attn`. Llama checkpoints may require Hugging Face access approval and authentication.

### 2. Enable Reflow (one block)

Add a `reflow` block to the `zero_optimization` section of a ZeRO Stage 3 config with CPU optimizer offload (the scripts generate this for you):

```jsonc
"offload_optimizer": {
    "device": "cpu",
    "pin_memory": true
},
"reflow": {}           // the only addition Reflow requires
```

An empty block is enough; every Reflow setting (NUMA core binding, thread counts) is optional — see [Configuration](#configuration). Remove the block to fall back to plain ZeRO-Offload.

### 3. Run a fine-tuning script

Each script takes the mode (`reflow` or `zerooffload`) as the first argument and an optional **global batch size** as the second (it must be divisible by the GPU count):

```bash
# bash <script> <reflow|zerooffload> [global_batch_size]
bash finetune_opt-30b_8gpu.sh   reflow        # OPT-30B, 8 GPUs
bash finetune_opt-30b_8gpu.sh   zerooffload   # ZeRO-Offload baseline
bash finetune_opt-30b_8gpu.sh   reflow 32     # override the batch size
bash finetune_llama-7b_1gpu.sh  reflow        # Llama-7B, 1 GPU
bash finetune_llama-8b_1gpu.sh  reflow        # Llama-8B, 1 GPU
bash finetune_llama-70b_8gpu.sh reflow        # Llama-70B, 8 GPUs
bash finetune_opt-350m-1gpu.sh  reflow        # OPT-350M, 1 GPU (smallest reproduction)
bash finetune_opt-30b_1gpu.sh   reflow        # OPT-30B, 1 GPU
```

`run_compare.sh` runs both modes of a script back to back and prints peak GPU/host memory, the
steady-state step time and the speedup:

```bash
bash run_compare.sh finetune_opt-350m-1gpu.sh
```

The generated config and training output directory are written next to the launch script. The comparison and equivalence checks put their logs in `compare_out/` and `bitexact_out/` by default (`OUT_DIR` overrides either). Scripts can be launched from another working directory.

The default run stops after 10 microsteps and excludes the first microstep from its mean time. For a longer run:

```bash
BENCH_STEPS=100 WARMUP_STEPS=10 ATTN_IMPLEMENTATION=sdpa \
  bash finetune_opt-350m-1gpu.sh reflow
```

`MAX_LENGTH`, `WORKER_CORES` (default 8), and `MAIN_CORES` (default 2 in these examples) also accept environment overrides. The runtime's main reservation default is 3; the scripts explicitly select 2. `gradient_clipping` is set to `0.0` in both modes for throughput comparison; enable clipping for training that requires it.

### Optimizer: Adam (default) or Lion

Every script defaults to Adam. The `_lion` scripts pass `--optimizer lion` and use `ReflowCPULion` (one momentum tensor instead of Adam's two — half the CPU optimizer state), and there are 4-GPU variants:

```bash
bash finetune_opt-30b_8gpu_lion.sh   reflow   # OPT-30B, 8 GPUs, Lion
bash finetune_opt-30b_4gpu.sh        reflow   # OPT-30B, 4 GPUs, Adam
bash finetune_opt-30b_4gpu_lion.sh   reflow   # OPT-30B, 4 GPUs, Lion
bash finetune_llama-70b_8gpu_lion.sh reflow   # Llama-70B, 8 GPUs, Lion
```

To use Lion in your own run, just add `--optimizer lion` to `finetune_zero3.py` — Reflow transparently remaps the CPU optimizer to its async subclass.

## Configuration

The scripts use `enable_cpu_affinity: true`, `bucketwise_cores_per_worker: 8`, and `main_thread_cores: 2`. CPU counts are logical by default. Workers are pinned to their allocated CPU masks; restricting the process to its rank slice does not by itself pin the main thread to the reserved CPUs.

Additional runtime options include `num_threads`, `state_update_cores`, `main_thread_core_type`, `pin_main_thread`, `worker_core_type`, `bucketwise_worker_affinity`, and `state_update_backward_cores`. Physical core reservations include SMT siblings; explicit main-thread pinning is optional. See the [Reflow configuration reference](https://github.com/deepspeedai/DeepSpeed/pull/8719) for the runtime change and its documentation. Standard ZeRO-3 knobs such as `sub_group_size` and `reduce_bucket_size` also apply.

## Verifying bit-level equivalence

Use deterministic attention and matching data order to compare the two paths. `--loss_check` enables deterministic algorithms and pins NCCL and cuBLAS settings before initialization, then logs each loss as a hexadecimal float. `--deterministic` enables deterministic algorithms without the additional comparison settings or loss logging.

```bash
bash check_bitexact.sh 1
OPTIMIZER=lion BENCH_STEPS=100 bash check_bitexact.sh 1
```

The checker generates matching CPU-offload configs, uses eager attention, and requires every requested step in both logs before reporting `BIT-IDENTICAL`. It exits unsuccessfully if either job fails, a loss log is incomplete, or any recorded loss differs. `BATCH_SIZE` is the **per-GPU microbatch size** in this checker (default 4); the model and optimizer are selected with `MODEL_NAME` and `OPTIMIZER` (default OPT-350M and Adam).

An exact loss match validates the recorded losses for that run. It does not establish bitwise equality of every optimizer state or guarantee equality for other hardware, library versions, or configurations. Deterministic comparison settings can reduce throughput; use ordinary runs for timing.

## Requirements

* A DeepSpeed build with Reflow, ZeRO Stage 3, and CPU optimizer offload (`offload_optimizer.device = "cpu"`).
* An x86 CPU with AVX (BF16 runs on both AVX-512 and AVX-256/AVX2). **AVX-512 is recommended** — AVX-256/AVX2 has no native BF16 support, so the whole BF16 path (gradient accumulation, FP32↔BF16 conversion, and the Adam updates) is emulated with extra instructions and can be slower. Enough cores per rank for the main thread plus the CPU-Adam workers.
* Host RAM for the CPU-resident FP32 master + optimizer state (e.g. Llama-70B: ~840 GB with Adam, ~560 GB with Lion — Lion keeps one momentum tensor instead of two).
* Reflow uses FP16/BF16 model parameters and gradients with FP32 master weights and optimizer states. The scripts in this directory select BF16. FP32 model parameters, low-precision master weights/states, and `fp32_optimizer_states: false` are rejected.

## Memory

Historical measurements on OPT-30B (8× B200, Adam, seq 512), steady-state during the training steps. These results have not been rerun with the current runtime revision:

| | GPU (8 GPUs, total) | Host RAM |
|---|---:|---:|
| ZeRO-Offload | 466 GB | ~625 GB |
| **Reflow** | 466 GB | ~549 GB |

* **GPU memory is identical** — the optimizer state and FP32 master live on the CPU either way, so Reflow adds no GPU cost.
* **Host RAM is ~76 GB lower with Reflow** (~12% on this run). ZeRO-Offload keeps a full CPU-side FP32 gradient buffer; in Reflow's bucketwise mode gradients are offloaded in BF16 and promoted to FP32 *inside* the CPU kernel, so that FP32 grad buffer is never held — only a compact half-precision grad buffer bounded by `sub_group_size` (not the full model) is kept.

**The bigger memory lever is the optimizer.** Lion keeps one momentum tensor, Adam two, so Lion needs **2/3 (−33%)** of the CPU optimizer state (FP32 master + moments; 4 bytes/param for the master and 4 per momentum):

| Model | FP32 master | Adam moments (2) | Lion moment (1) | Adam optimizer state | Lion optimizer state |
|---|---:|---:|---:|---:|---:|
| OPT-30B   | 120 GB | 240 GB | 120 GB | **~360 GB** | **~240 GB** |
| Llama-70B | 280 GB | 560 GB | 280 GB | **~840 GB** | **~560 GB** |

So for host-RAM-bound setups, Reflow + **Lion** is the lightest option: half-precision gradients (no FP32 grad buffer) plus a single momentum — Llama-70B fits its optimizer state in ~560 GB where Adam needs ~840 GB, and Lion is also the faster of the two at that scale (see [Performance](#performance)).

## Performance

Historical steady-state per-step time (avg of steps 2–10), Reflow vs the ZeRO-Offload baseline, on B200 GPUs, seq 2048, BF16. These measurements used earlier CPU placement settings and have not been rerun with the current revision or the default of 8 CPUs per worker. The speedup grows as the **per-rank optimizer shard** grows (bigger model or fewer GPUs) — there is more CPU optimizer work to hide inside the backward. Lion (one momentum) is lighter than Adam (two) and pulls ahead at scale.

| Model | GPUs | Optimizer | Reflow | ZeRO-Offload | **Speedup** |
|---|:---:|:---:|---:|---:|:---:|
| Llama-8B  | 1 | Adam | 1474 ms | 5945 ms | **4.03×** |
| Llama-13B | 1 | Adam | 1703 ms | 7676 ms | **4.51×** |
| OPT-30B   | 4 | Adam | 3070 ms | 6801 ms | **2.22×** |
| OPT-30B   | 4 | Lion | 3072 ms | 6538 ms | **2.13×** |
| OPT-30B   | 8 | Adam | 3111 ms | 5898 ms | **1.90×** |
| OPT-30B   | 8 | Lion | 3086 ms | 5627 ms | **1.82×** |
| Llama-70B | 8 | Adam | 4190 ms | 9884 ms | **2.36×** |
| Llama-70B | 8 | Lion | 3327 ms | 8701 ms | **2.61×** |

At 70B, Lion is ~20% faster than Adam (3327 vs 4190 ms) and needs ~560 GB of CPU state vs Adam's ~840 GB, because its single momentum halves the offloaded optimizer state.

### Example output

OPT-30B, 8× B200, seq 2048, BF16. Each step logs `TFLOPS` (counting the activation-recompute forward) and `effective TFLOPS` (without it).

**Reflow** (`bash finetune_opt-30b_8gpu.sh reflow`):

```
Reflow CPU-Adam SIMD: AVX-512
Step    1 | Loss: 2.3329 | Time:  7712ms | TFLOPS(w/ recompute):  4267.00 | effective TFLOPS(w/o recompute):  3200.25 | Tokens/s:   2124
Step    2 | Loss: 1.6865 | Time:  3110ms | TFLOPS(w/ recompute): 10580.75 | effective TFLOPS(w/o recompute):  7935.57 | Tokens/s:   5268
Step    3 | Loss: 1.7266 | Time:  3098ms | TFLOPS(w/ recompute): 10621.35 | effective TFLOPS(w/o recompute):  7966.01 | Tokens/s:   5288
Step    4 | Loss: 1.4287 | Time:  3108ms | TFLOPS(w/ recompute): 10589.88 | effective TFLOPS(w/o recompute):  7942.41 | Tokens/s:   5272
Step    5 | Loss: 1.3309 | Time:  3121ms | TFLOPS(w/ recompute): 10545.24 | effective TFLOPS(w/o recompute):  7908.93 | Tokens/s:   5250
Step    6 | Loss: 1.3002 | Time:  3126ms | TFLOPS(w/ recompute): 10527.12 | effective TFLOPS(w/o recompute):  7895.34 | Tokens/s:   5241
Step    7 | Loss: 1.6794 | Time:  3117ms | TFLOPS(w/ recompute): 10558.52 | effective TFLOPS(w/o recompute):  7918.89 | Tokens/s:   5257
Step    8 | Loss: 1.5415 | Time:  3108ms | TFLOPS(w/ recompute): 10588.21 | effective TFLOPS(w/o recompute):  7941.16 | Tokens/s:   5272
Step    9 | Loss: 1.5562 | Time:  3120ms | TFLOPS(w/ recompute): 10547.18 | effective TFLOPS(w/o recompute):  7910.39 | Tokens/s:   5251
Step   10 | Loss: 1.4714 | Time:  3153ms | TFLOPS(w/ recompute): 10436.84 | effective TFLOPS(w/o recompute):  7827.63 | Tokens/s:   5196
```

**ZeRO-Offload** (`bash finetune_opt-30b_8gpu.sh zerooffload`):

```
Step    1 | Loss: 2.3329 | Time:  8753ms | TFLOPS(w/ recompute):  3759.61 | effective TFLOPS(w/o recompute):  2819.71 | Tokens/s:   1872
Step    2 | Loss: 1.6855 | Time:  5873ms | TFLOPS(w/ recompute):  5603.57 | effective TFLOPS(w/o recompute):  4202.68 | Tokens/s:   2790
Step    3 | Loss: 1.7261 | Time:  5873ms | TFLOPS(w/ recompute):  5602.89 | effective TFLOPS(w/o recompute):  4202.16 | Tokens/s:   2789
Step    4 | Loss: 1.4260 | Time:  5860ms | TFLOPS(w/ recompute):  5616.12 | effective TFLOPS(w/o recompute):  4212.09 | Tokens/s:   2796
Step    5 | Loss: 1.3316 | Time:  5799ms | TFLOPS(w/ recompute):  5675.13 | effective TFLOPS(w/o recompute):  4256.35 | Tokens/s:   2825
Step    6 | Loss: 1.2991 | Time:  5892ms | TFLOPS(w/ recompute):  5585.03 | effective TFLOPS(w/o recompute):  4188.77 | Tokens/s:   2781
Step    7 | Loss: 1.6796 | Time:  5942ms | TFLOPS(w/ recompute):  5537.87 | effective TFLOPS(w/o recompute):  4153.40 | Tokens/s:   2757
Step    8 | Loss: 1.5477 | Time:  5947ms | TFLOPS(w/ recompute):  5533.28 | effective TFLOPS(w/o recompute):  4149.96 | Tokens/s:   2755
Step    9 | Loss: 1.5629 | Time:  5871ms | TFLOPS(w/ recompute):  5605.00 | effective TFLOPS(w/o recompute):  4203.75 | Tokens/s:   2791
Step   10 | Loss: 1.4747 | Time:  5790ms | TFLOPS(w/ recompute):  5683.77 | effective TFLOPS(w/o recompute):  4262.83 | Tokens/s:   2830
```

Reflow is ~1.9× faster (≈3.1 s vs ≈5.9 s per step) by overlapping the CPU-Adam step with the backward pass.

## Notes

* **NUMA core binding**: every launch script passes `--bind_cores_to_rank`. Reflow allocates its main reservation and worker masks within the rank's CPU slice. Use physical core reservations and `pin_main_thread` if you need explicit separation of the main thread and worker SMT siblings.
* Use `engine.step()` after `deepspeed.initialize()`; direct calls to `ReflowCPUAdam.step()` or `ReflowCPULion.step()` are rejected.
* `--save_checkpoint` requires participation by every rank. The script saves the tokenizer on rank 0 and propagates checkpoint failures.
* **AVX-512 is strongly recommended** for the CPU optimizer; on AVX2 the BF16 path is emulated and slower (see [Requirements](#requirements)).

## Related resources

* [DeepSpeed ZeRO-Offload tutorial](https://www.deepspeed.ai/tutorials/zero-offload/)
* [ZeRO Stage 3 config options](https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training)
* Reflow runtime module: `deepspeed/runtime/reflow/` — `reflow_stage3.py`, `reflow_cpu_adam.py`, `reflow_cpu_lion.py`

## Validation

Run the CPU launcher and data-preprocessing checks without initializing CUDA:

```bash
CUDA_VISIBLE_DEVICES='' DS_ACCELERATOR=cpu python3 -m unittest discover -s tests -v
```

These checks validate launcher arguments, generated configurations, failure propagation, loss-log completeness, and padding masks. They do not replace GPU training validation. Use `check_bitexact.sh` on available GPUs to compare training losses and `run_compare.sh` to measure throughput. GPU memory sampling reports device usage, including unrelated jobs; host memory sampling sums RSS of matching training processes and can overcount shared pages.
