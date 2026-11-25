# Benchmark plan: rdsp vs NVIDIA Megatron pipeline parallelism

## Next phase (2026-10-03): placement-matched comparison with MegatronMIMO

MegatronMIMO (Megatron-LM, arXiv 2605.27678) gives the vision encoder and the
LLM separate layouts. Every run below has the same GPU layout on both sides;
`bench/mimo/runs.sh` holds the exact commands.

| Run | Megatron-Bridge | rdsp |
|---|---|---|
| shared | standard recipe, TP2·PP2·DP2, encoder on PP0 | 2 stages TP2×DP2, cut 16, encoder on stage 0 |
| non-colocated | MIMO: images DP2 (a row each) + language TP2·PP2, 6 GPUs | vision stage DP2 + 2 TP2 stages, 6 GPUs |
| colocated | not available for Qwen3.5 (Bridge: coming soon) | `ColocatedVision()`, language as in shared |
| best | small sweep per system | small sweep, per-stage layouts allowed |

Held identical (Megatron defaults overridden where they differ):
- Qwen3.5-9B from the same HF checkpoint, all parameters trained, MTP head off.
- CORD-v2 through Bridge's own pipeline (`export_bridge_batches.py` writes its
  samples; rdsp reads them in order): same prompt, image pixel limits
  (200,704-1,003,520), tokens and supervised positions. Sequential sampler
  (`dataloader_type=single`) on both Megatron paths.
- seq 2048, global batch 64, 2-row microbatches, padding to a multiple of
  128 (Megatron per microbatch, rdsp per step, so rdsp never pads less).
- Loss: mean over a rank's supervised tokens per microbatch, then over
  microbatches and ranks: Megatron's default (`calculate_per_token_loss=False`,
  which the 9B recipe and MIMO use), rdsp `--loss microbatch-mean`.
- AdamW lr 1e-5 constant, betas 0.9/0.95, eps 1e-8, no weight decay, no
  warmup, no clipping; optimizer state sharded over data parallel
  (distributed optimizer / ZeRO-1); bf16 with fp32 master weights and fp32
  gradient reduction. No recompute, no offload.

Gate before timing: Megatron and rdsp report the same step-0 loss in the
shared layout; rdsp with a one-rank last stage matches the unsplit HF model.

Reported: median step time (steps 6-50), real (unpadded) tokens/s and per
active GPU, $/M tokens, peak memory per GPU, the 50-step loss curves.
Differences that stay: Transformer Engine vs Hugging Face kernels, Megatron's
on-the-fly data workers vs rdsp's prepared batches; each system is also
reported against its own shared run.

## Previous phase (2026-09-26): 30B vision-language on one node

The Qwen3-0.6B benchmark below is finished (results in `docs/BENCHMARK_RESULTS.md`).
Following review feedback (target a ~30B VL model, non-NVLink GPUs), the current
phase changes rdsp source, CPU-tested first, in this order:

1. **Per-stage weight loading.** `rdsp.initialize(weights=<HF dir>)`: the driver
   model may be a meta-device skeleton; each stage reads only its own tensors
   from the safetensors files.
2. **HF stage builder** (`hf_stage.py`): every stage runs the model's own
   forward; blocks outside its range pass through, unowned modules are Identity.
3. **Multi-tensor boundaries**: the hidden state plus the tensor keyword
   arguments the model passes to its blocks (e.g. Qwen3-VL rotary tables).
4. **VL data path**: stage-0 inputs may be a dict; non-row-shaped values
   (pixels, image grids) are given per row and concatenated per rank.
5. Later: optimizer offload, activation recompute, cost-balanced cuts; then GPU
   runs (Qwen3-VL-2B/8B on 4xL4, Qwen3-VL-32B on 8xL40S). Multi-node is out of scope.

Tests: tiny Qwen3-VL on Ray CPU actors must match the unsplit model's losses.

---

Status of the original benchmark plan below: **plan only** when written. Written 2026-09-22.

How each claim was checked:

- **[src]**: read in source code. Megatron-LM at tag `core_v0.19.2` (commit `4b4acac`, 2026-09-18). Megatron-Bridge at tag `v0.6.2` (commit `c0e164e`, 2026-09-18). DeepSpeed at the pinned oracle revision `53a2ac4` (the local `DeepSpeed/` checkout). rdsp as it is in this repo today.
- **[doc]**: stated in official docs or release notes. URLs are in §11.
- **[calc]**: computed here. The arithmetic is shown.
- **[unverified]**: not checked. Treat it as a hypothesis and confirm it in the smoke test (§8, Tier 0).

---

## 0. Summary

- **Comparator.** Use Megatron-Core **0.19.2** (the latest release) through Megatron-LM's `pretrain_gpt.py` for "Megatron as people run it". Use a short custom **Megatron-Core training loop that loads the Hugging Face (HF) weights through Megatron-Bridge 0.6.2 (`AutoBridge`)** for loss-parity runs and as a cross-check. Both run in the NGC container `nvcr.io/nvidia/pytorch:26.01-py3`, which is the tag Megatron's own install guide uses.
- **Same software stack for both frameworks.** Install rdsp, Ray 2.58, and the pinned DeepSpeed revision into the **same** NGC image. Then torch, CUDA, cuBLAS and NCCL are identical on both sides.
- **Transformer Engine (TE) is effectively required for a fair Megatron run.** `--transformer-impl local` exists [src], but it swaps in unfused attention that materializes the full s×s score matrix. Megatron-Bridge's Qwen3 weight names also assume the TE layer layout [src]. Use TE, turn off Megatron's optional fusions for the "matched" row, and report "Megatron defaults" as a separate row.
- **Normalize out kernel differences.** Measure a 1-GPU baseline for each framework and report *pipeline efficiency* = throughput at p stages ÷ (p × that framework's own 1-GPU throughput). This isolates the pipeline runtime, which is what rdsp changes, from kernel quality, which rdsp does not change.
- **Expect a large bubble on both frameworks.** The LM head is 22–25% of the forward FLOPs and sits on the last stage. With uniform layer splits, the last stage does 1.6× (PP=2) to 2.3× (PP=4) the work of the others [calc]. So the idle-time fraction will be far above the textbook (p−1)/(m+p−1). Both frameworks share this. A "balanced split" row (rdsp `ExplicitCuts` vs Megatron `--pipeline-model-parallel-layout`) is included.
- **Two findings about rdsp came out of planning this:**
  1. With `StageOverride(num_gpus=k)`, every rank of a stage receives the **same** microbatch (`StageGroupClient.submit` sends identical `inputs`/`upstream` to all ranks and forwards rank 0's result) [src]. Extra GPUs on a stage today **replicate** its work rather than share it. So the heterogeneous "capability" row can show that the configuration is expressible and what it costs, but **not** a speedup until microbatches are split within a stage (§4.4).
  2. DeepSpeed clips gradients **per stage** (default `gradient_clipping` = 1.0 at the pinned revision [src]). Megatron clips on the **global** norm across stages. The two are not the same algorithm. Turn clipping off on both sides for the benchmark (§2).
- **Cost** [calc, rough]: about **$95** of Modal GPU+CPU time for the full planned matrix with 3 repeats. Budget **$150–200** including smoke tests and debugging (§7.4).

---

## 1. How to run Qwen3-0.6B with Megatron pipeline parallelism (September 2026)

### 1.1 The options

| Path | What it is | Use it for |
|---|---|---|
| **Megatron-LM `pretrain_gpt.py` + CLI flags** | Megatron's reference training script. Qwen3 is expressed with flags (below). Mock data. | Throughput rows "Megatron-plain" and "Megatron-defaults". This is the path most published Megatron numbers use. |
| **Megatron-Core API in a custom loop + Megatron-Bridge `AutoBridge`** | `AutoBridge.from_hf_pretrained(...)` → `to_megatron_provider(load_weights=True)` → set PP → `provide_distributed_model()`. Then drive `get_forward_backward_func()` directly [src: `examples/conversion/hf_megatron_roundtrip_multi_gpu.py`, `examples/run_simple_mcore_train_loop.py`]. | Loss-trajectory parity from identical HF weights. Throughput cross-check. Exact control over the data. |
| Megatron-Bridge recipe runner (`scripts/training/train.sh`, recipe `qwen3_600m_pretrain_config`) | NVIDIA's recommended path for HF models. The recipe exists [src: `recipes/qwen/h100/qwen3.py`]. | Not used. The launcher is built around Slurm and NeMo-Run [doc: `scripts/training/README.md`], which is awkward on Modal. It also turns on tuned defaults (for example `cross_entropy_fusion_impl="te"` and manual GC) that you would have to undo one by one. |

Versions to pin:

- Megatron-Core **0.19.2**: PyPI `megatron-core==0.19.2`, requires Python ≥3.12; git tag `core_v0.19.2` [doc: PyPI; src: tag list]. `main` is 0.20.0-dev [src: `megatron/core/package_info.py`].
- Megatron-Bridge **0.6.2**: PyPI `megatron-bridge==0.6.2`, requires Python 3.12, `transformers>=5.8,<=5.12.1` [doc: PyPI; src: `pyproject.toml`]. The rdsp dev venv has transformers 5.16.1, so **pin transformers 5.12.1 in the shared image** and re-run the P6 test on it.
- NGC container: Megatron's install guide uses `nvcr.io/nvidia/pytorch:26.01-py3` and says to prefer "the previous month's" container over the newest [src: `docs/get-started/install.md`]. 26.01 ships CUDA 13.1.1, PyTorch 2.10.0a0 and TE 2.11 [doc: release notes 26.01]. Megatron CI uses 26.04 and 26.06 [src: `docker/Dockerfile.ci.*`]. Bridge CI uses 26.06 [src].
- **Driver caveat.** Modal hosts run driver 580.95.05, which is CUDA driver API 13.0 [doc: Modal CUDA guide]. Modal only guarantees CUDA versions no newer than the host. 26.01 (CUDA 13.1) would rely on CUDA minor-version compatibility. **[unverified]** on Modal. Fallback: `25.10-py3` (CUDA 13.0.2 [doc]) with the matching Megatron-Core of that month. The Tier 0 smoke test decides.

### 1.2 Qwen3-0.6B as Megatron flags

HF config [doc: `Qwen/Qwen3-0.6B/config.json`]: 28 layers, hidden 1024, FFN 3072, 16 query heads, 8 KV heads, **head_dim 128** (so the query projection is 16×128 = 2048 wide, which is *not* equal to hidden), RMSNorm eps 1e-6, RoPE θ = 1e6, vocab 151936, `tie_word_embeddings: true`, no attention bias, SiLU gated MLP.

```
--num-layers 28 --hidden-size 1024 --ffn-hidden-size 3072
--num-attention-heads 16 --group-query-attention --num-query-groups 8
--kv-channels 128                      # head_dim; required because 16*128 != 1024
--qk-layernorm                         # Qwen3 q_norm / k_norm
--normalization RMSNorm --norm-epsilon 1e-6
--swiglu --disable-bias-linear
--position-embedding-type rope --rotary-base 1000000 --rotary-percent 1.0
--untie-embeddings-and-output-weights  # see "tied embeddings" below
--max-position-embeddings 40960
--attention-dropout 0.0 --hidden-dropout 0.0 --init-method-std 0.02
--vocab-size 151936 --make-vocab-size-divisible-by 128   # 151936 = 1187*128, so no padding
--tokenizer-type NullTokenizer --mock-data
```

Notes:

- Flag names were checked against `core_v0.19.2` [src]. Many flags are now generated from dataclass fields: `--qk-layernorm`, `--transformer-impl`, `--attention-backend`, `--num-query-groups` and `--kv-channels` come from `TransformerConfig` via `ArgumentGroupFactory` (`megatron/training/argument_utils.py`). `--norm-epsilon` is an explicit alias of `layernorm_epsilon`. The smoke test should still run `python pretrain_gpt.py --help | grep -E 'qk-layernorm|kv-channels|transformer-impl'`.
- `NullTokenizer(vocab_size)` has vocab exactly `vocab_size`, with EOD = `vocab_size-1` [src: `megatron/core/tokenizers/text/libraries/null_tokenizer.py`]. Check the "padded vocab" line in the log reads 151936.
- **Tied embeddings.** rdsp refuses to split a tied embedding/head pair across stages. So it unties by cloning the embedding into `lm_head` (`tests/integration/test_p6_first_row.py::load_untied`). Megatron *can* keep them tied across stages: it keeps a copy on the last stage and all-reduces the two gradients. That is a real difference in work: an extra all-reduce of a 155.6M-parameter gradient per step. For apples-to-apples, **untie on both sides** (`--untie-embeddings-and-output-weights`). Megatron's tied behaviour can appear as an extra row inside "Megatron-defaults" if wanted.
- Qwen3 differs from Qwen2 only in having no QKV bias and in adding QK-norm. That matches the Bridge's `Qwen3Bridge.provider_bridge` [src: `megatron/bridge/models/qwen/qwen3_bridge.py`].

### 1.3 Running without Transformer Engine (`--transformer-impl local`): possible, not recommended

- `transformer_impl: Literal['local','transformer_engine','inference_optimized'] = "transformer_engine"` [src: `transformer_config.py:1220`]. So `--transformer-impl local` is valid.
- What "local" gives you [src: `gpt_builders.py::_get_transformer_layer_spec`, `gpt_layer_specs.py`, `models/backends.py`]:
  - RMSNorm becomes `WrappedTorchNorm` (torch's RMSNorm). Apex `FusedLayerNorm` is used only for LayerNorm, and only if apex is installed.
  - Attention becomes `DotProductAttention`: unfused, and it **materializes the full s×s score matrix**. At s=2048 that is 16×2048²×2 B = 128 MiB per layer per microbatch just for the bf16 scores [calc]. That is slow, and at PP=1 or PP=2 on a 24 GB L4 it risks running out of memory.
  - Linears become Megatron's own `ColumnParallelLinear`/`RowParallelLinear`. The `gradient_accumulation_fusion` path needs apex's `fused_weight_gradient_mlp_cuda` extension (`tensor_parallel/layers.py:51,1073`), so pass `--no-gradient-accumulation-fusion` if apex is absent.
  - `--attention-backend local` additionally requires `--spec local` (`arguments.py:435`).
  - The optimizer falls back to `torch.optim.AdamW` if neither TE nor apex imports (`megatron/core/optimizer/__init__.py:13-33`).
- **Megatron-Bridge weight loading assumes the TE layer layout.** The Qwen3 mapping uses TE's fused names, for example `self_attention.linear_qkv.layer_norm_weight` and `mlp.linear_fc1.layer_norm_weight` [src: `qwen3_bridge.py`]. The local spec has separate `input_layernorm` / `pre_mlp_layernorm` modules. Loading HF weights into a local-spec model through Bridge will likely miss those tensors. **[unverified]**: not run. Treat it as unsupported.
- **Pip install without NGC.** `megatron-core` alone installs without TE (TE is the optional `[te]` extra: `transformer-engine[pytorch,core_cu13]`) [src: `pyproject.toml`]. Building TE's torch extension from source takes a long time and needs matching CUDA headers. The NGC image avoids all of this. **Recommendation: NGC + TE.** Keep `local` only as an optional "least-fused Megatron" row at s=512 on H100.

### 1.4 Minimal launches

Plain 1F1B (the same schedule rdsp uses: stage i runs `p-1-i` warm-up forwards, then alternates one forward and one backward [src: `schedule.py::_ops_1f1b` and `schedules.py::forward_backward_pipelining_without_interleaving`]):

```bash
# PP=2, 8 microbatches, seq 2048, micro-batch 1  -> global batch 8
torchrun --nproc_per_node 2 pretrain_gpt.py <QWEN3 FLAGS> \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 2 \
  --seq-length 2048 --micro-batch-size 1 --global-batch-size 8 ...

# PP=4, 16 microbatches, seq 512, micro-batch 4 -> global batch 64
torchrun --nproc_per_node 4 pretrain_gpt.py <QWEN3 FLAGS> \
  --pipeline-model-parallel-size 4 \
  --seq-length 512 --micro-batch-size 4 --global-batch-size 64 ...
```

With data parallel size 1, microbatches per step = global batch ÷ micro batch.

**Interleaved 1F1B** (virtual pipeline stages, abbreviated VPP; an extra Megatron-only comparison). Megatron splits the 28 layers evenly by default: embedding on stage 0, head and loss on the last stage.

- PP=2: 14 layers per rank → `--num-virtual-stages-per-pipeline-rank 2` (7 layers per chunk).
- PP=4: 7 layers per rank, which is prime, so `--num-layers-per-virtual-pipeline-stage` can only be 1 or 7. Use an explicit layout with 8 chunks that still puts 7 layers on each rank. Chunk *j* lives on rank *j mod 4*: `--pipeline-model-parallel-layout "Et*4|t*3|t*3|t*4|t*3|t*4|t*4|t*3,L"` gives ranks 4+3, 3+4, 3+4, 4+3 = 7 each. The `x*n` syntax is from `pipeline_parallel_layer_layout.py::parse_str_to_list` [src]. **[unverified]**: the TransformerConfig layout validator has not been run on this string.
- The interleaved schedule needs `num_microbatches ≥ pipeline size` (default `microbatch_group_size_per_vp_stage` = PP [src: `schedules.py:1134`]). All m ∈ {4, 8, 16} qualify.
- `--no-overlap-p2p-communication` only affects the interleaved schedule [src: `arguments.py:2965`]. Leave overlap **on** in the "defaults" row and **off** in the "plain" row.

The full script is in §9.1.

---

## 2. Fairness controls

### 2.1 Controls held identical

| Control | rdsp setting | Megatron setting | Why |
|---|---|---|---|
| Model | Qwen3-0.6B, untied (lm_head = clone of embed), bf16 | Same config (§1.2), `--untie-embeddings-and-output-weights`, `--bf16` | Same work |
| Stage split | `UniformTransformerBlocks`: 14/14 or 7/7/7/7; embed on first stage, norm+head on last [src: `partition.py`] | Megatron default even split; embed first, head+loss last | Same imbalance (§3.4) |
| Schedule | 1F1B | non-interleaved 1F1B | Same algorithm |
| Tokens per microbatch | 2048: `(b, s)` = (4, 512) or (1, 2048) | `--micro-batch-size b --seq-length s` | Same per-kernel shapes |
| Microbatches m | `PipelineConfig(microbatches=m)` = DS `gradient_accumulation_steps` | `--global-batch-size b*m` | Same bubble |
| Data | Synthetic token ids from one seeded generator. Labels pre-shifted: input = x[:, :s], label = x[:, 1:s+1] | Custom loop: the same tensors. `pretrain_gpt.py`: `--mock-data` (same shapes, different ids) | Shapes are what matter for speed. Parity runs use identical ids. |
| Loss | Mean token cross-entropy in fp32 on the last stage, averaged over m | Megatron vocab-parallel CE in fp32, mean over tokens, schedule divides by m | Same math |
| Optimizer | AdamW, lr 1e-5 constant, betas (0.9, 0.999), eps 1e-8, **weight decay 0**, fp32 master weights | `--optimizer adam` (AdamW mode is the default, `decoupled_weight_decay=True` [src]), same hyperparameters, `--lr-decay-style constant --lr-warmup-iters 0` | |
| Grad clipping | **`"gradient_clipping": 0.0`**. DeepSpeed's default is 1.0 [src: `constants.py:254`] and applies per stage. | **`--clip-grad 0.0`** (default 1.0, global norm across stages) | The two clip different quantities, so turn both off |
| Grad accumulation dtype | ZeRO-0 + bf16 selects DeepSpeed's `DDP_BFLOAT16` → `FP16_Optimizer` with bf16 grads, so microbatch grads accumulate in **bf16** `param.grad` [src: `engine.py:1883-1891`] | "plain": **`--grad-reduce-in-bf16`** (main-grad buffer in bf16 [src: `arguments.py:1186-1193`]). "defaults": fp32 main grads. | Match what rdsp does today. See §2.3 for the fp32 option. |
| Activation recompute | none (rdsp has none) | no `--recompute-*` flags (default off) | |
| Dropout | 0 (Qwen3 config) | 0 | |
| Distributed optimizer, overlapped grad reduce | n/a (1 GPU per stage) | off (no-ops at data parallel size 1, but they change code paths) | |
| CUDA graphs, torch.compile, FP8 | off | off (`cuda_graph_impl` default "none") | |
| GPUs | same Modal GPU type and count; `H100!` to stop Modal's silent H100→H200 upgrade [doc: Modal GPU guide] | same | |
| Software | same NGC image: torch, CUDA, NCCL, cuBLAS | same | |
| Warm-up | first 10 steps dropped (NCCL communicator setup, cuBLAS heuristics, allocator growth, Ray worker warm-up) | same | |
| Run order | cells of the two systems alternated (A B A B) inside one container | same | Cancels drift (thermals, noisy neighbours) |

### 2.2 Differences that stay, and are documented rather than removed

- **Layer kernels.** rdsp runs HF `Qwen3DecoderLayer` in PyTorch eager mode with SDPA attention (HF's default `attn_implementation`), which picks FlashAttention or memory-efficient kernels. HF also repeats the KV heads before SDPA. Megatron-TE fuses RMSNorm into the following linear (`TELayerNormColumnParallelLinear`), and that fusion cannot be turned off inside the TE spec. Mitigations:
  - (a) `--attention-backend flash`, the closest match to SDPA-flash. If TE cannot find a flash backend, use `fused` (cuDNN) and record which one was used. **[unverified]** whether NGC 26.01 ships the `flash-attn` package TE needs for the `flash` backend.
  - (b) Turn off every optional Megatron fusion in the "plain" row (below).
  - (c) Report kernel-normalized pipeline efficiency (§3.1).
- **Optimizer kernel.** Megatron uses TE `FusedAdam` when TE imports [src: `optimizer/__init__.py:14`]. For rdsp, set `"torch_adam": true, "fused": true` so DeepSpeed builds `torch.optim.AdamW(fused=True)` (DeepSpeed passes extra params through [src: `engine.py:1974-1977`]). Both are single multi-tensor kernels. **[unverified]**: that `fused=True` is accepted under DeepSpeed's `FP16_Optimizer` wrapper for bf16. Also report optimizer time separately on both sides.
- **Stage-to-stage transport.** This is *the thing being measured*: Ray object store (or RDT) vs Megatron's NCCL `batch_isend_irecv`. On H100 NVLink it strongly favours Megatron. On L4 (PCIe, no NVLink) NCCL may itself go through host memory. Record `nvidia-smi topo -m` and `NCCL_DEBUG=INFO` transport lines ("via P2P/IPC", "SHM", "NET") in every container.
- **Driver-side control.** rdsp's driver dispatches every command through Ray and blocks on the ready/apply barrier each step. Megatron ranks run their schedule locally. This is also part of what is being measured.

### 2.3 Megatron variants

| Variant | Flags beyond §1.2 | Purpose |
|---|---|---|
| **M-plain** (headline comparator) | `--transformer-impl transformer_engine --attention-backend flash --no-gradient-accumulation-fusion --no-bias-swiglu-fusion --no-rope-fusion --no-bias-dropout-fusion --no-persist-layer-norm --no-masked-softmax-fusion --grad-reduce-in-bf16 --clip-grad 0.0` | As close to rdsp as Megatron allows |
| **M-defaults** | TE, `--attention-backend auto`, default fusions on, fp32 main grads, `--cross-entropy-loss-fusion`, `--manual-gc`, `CUDA_DEVICE_MAX_CONNECTIONS=1` (as in `examples/llama/*.sh`), `--clip-grad 0.0` | "What a Megatron user gets" |
| **M-interleaved** | M-defaults + VPP (§1.4) | Megatron-only schedule, extra row |
| M-local (optional) | `--transformer-impl local --no-gradient-accumulation-fusion`, s=512, H100 only | Least-fused Megatron |

**fp32 gradient accumulation (optional check).** To give rdsp fp32 accumulation like Megatron's default, DeepSpeed needs ZeRO-1 + `"data_types": {"grad_accum_dtype": "fp32"}`, which selects `BF16_Optimizer` [src: `engine.py:1863-1868`]. ZeRO-0 + bf16 + fp32 accumulation raises `NotImplementedError` [src: `engine.py:1893`]. Whether `BF16_Optimizer` accumulates correctly under rdsp's "gradient_accumulation_steps=1, many backward calls, one step" pattern is **[unverified]**. Gate this option behind a gradient-parity test like P2's before using it. Do not use it in the headline rows.

---

## 3. Metrics and how to compute them

### 3.1 Step time, throughput, pipeline efficiency

- **Step time.** Wall time of one optimizer step.
  - rdsp: driver-side `perf_counter` around `engine.train_batch()`. It returns after every stage's apply returned [src: `coordinator.py::_run`]. `engine.step()` is launched asynchronously, so the last kernel tail spills into the next step. That is fine in steady state.
  - Megatron `pretrain_gpt.py`: "elapsed time per iteration (ms)" with `--log-interval 1`.
  - Custom Megatron loop: `perf_counter` around `forward_backward_func + optimizer.step()` on the last rank.
  - Report the **median and p90 over N = 50 steps after 10 warm-up steps**, plus total time ÷ N (robust against per-step boundary jitter).
- **tokens/s** = b·s·m ÷ step time. **tokens/s/GPU** divides by the number of GPUs *allocated* (5 in the capability row).
- **Pipeline efficiency** E = tokens/s(PP=p) ÷ (p × tokens/s(PP=1, same framework, same b, s, m)). The PP=1 baselines are Megatron PP=1, plain DeepSpeed on 1 GPU with the same stage module, and rdsp `stages=1` (which adds Ray dispatch but no pipelining). E removes kernel differences. **Report E and the absolute numbers side by side.**

### 3.2 MFU (model FLOPs utilization)

Use Megatron's own convention [src: `training.py::num_floating_point_operations`]. Count 3× the forward GEMMs (forward, weight-gradient, data-gradient). Count 2 FLOPs per multiply-add. Count core attention at **half** (causal mask). Ignore norms, softmax and elementwise work. Embedding lookup is 0 FLOPs.

For Qwen3-0.6B with h=1024, L=28, q = n_heads·head_dim = 2048, kv = 8·128 = 1024, ffn = 3072, V = 151936:

```
per layer, per token, forward:
  QKV + out proj : 2·[h·(q + 2·kv) + q·h] = 2·6,291,456 = 12.58 MFLOP
  gated MLP      : 2·3·h·ffn              = 18.87 MFLOP
  core attention : 2·2·q·s / 2 (causal)   = 4096·s FLOP
LM head          : 2·h·V                  = 311.16 MFLOP
training FLOPs per token  F(s) = 3·[ 28·31.457M + 311.16M + 28·4096·s ]
                              = 3·[ 880.80M + 311.16M + 114,688·s ]
  F(512)  = 3.752 GFLOP/token   (non-causal accounting: 3.928)
  F(2048) = 4.281 GFLOP/token   (non-causal accounting: 4.985)
MFU = F(s) · tokens/s / (N_GPUs · peak_bf16_dense)
```

[calc]. Parameter check: blocks 440.4M, embedding 155.6M, untied total 751.6M, tied 596.0M, which matches "0.6B" [calc]. The LM head is 24.9% (s=512) and 21.8% (s=2048) of forward FLOPs.

Peak dense bf16: H100 SXM **989 TFLOPS**; L4 **121 TFLOPS** (the datasheet's 242 is "with sparsity") [doc: NVIDIA datasheets, **verify at write-up**]. The L4 is a 72 W card and sustains well below its boost-clock peak. Also report the achieved SM clock (`nvidia-smi --query-gpu=clocks.sm`).

Compute MFU with **this one formula for both frameworks**. Use Megatron's printed "TFLOP/s/GPU" (`--log-throughput`) only as a cross-check.

### 3.3 Pipeline bubble (idle-time fraction)

Measured, per stage: `bubble_i = 1 − busy_i / step_time`, where busy_i is GPU time spent in forward and backward compute on stage i during the step. Optimizer time is reported separately and excluded from busy. Report the mean over stages and the worst stage.

- rdsp: busy_i = sum of CUDA-event durations from the engine proxy (§5.1).
- Megatron: busy_i = `forward-compute` + `backward-compute` timers at `--timing-log-level 2 --timing-log-option all` [src: `schedules.py:488,545`; `training.py:2640-2652`].
- Megatron timers synchronize the GPU. So take bubble and communication numbers from **separate instrumented runs**, and take throughput from clean runs with `--timing-log-level 0`.

Theoretical, two versions:

- Balanced stages: `(p−1)/(m+p−1)` (Narayanan et al., SC'21).
- **Imbalance-aware**, with per-stage fwd+bwd time t_i: `T ≈ Σ t_i + (m−1)·max t_i`, and `bubble = 1 − m·Σt_i/(p·T)`. Plug in the measured t_i. Values from FLOPs-proportional t_i [calc]:

| p | m | balanced | imbalanced s=512 | imbalanced s=2048 | ideal speed-up vs 1 GPU (s=2048) |
|---|---|---|---|---|---|
| 2 | 4 | 0.200 | 0.304 | 0.293 | 1.41 |
| 2 | 8 | 0.111 | 0.255 | 0.240 | 1.52 |
| 2 | 16 | 0.059 | 0.228 | 0.211 | 1.58 |
| 4 | 4 | 0.429 | 0.567 | 0.554 | 1.79 |
| 4 | 8 | 0.273 | 0.507 | 0.487 | 2.05 |
| 4 | 16 | 0.158 | 0.470 | 0.445 | 2.22 |

### 3.4 Per-stage load

Forward MFLOP per token by stage, uniform split [calc]:

| s | PP=2 | PP=4 |
|---|---|---|
| 512 | 470 / 781 (1.66×) | 235 / 235 / 235 / 546 (2.32×) |
| 2048 | 558 / 869 (1.56×) | 279 / 279 / 279 / 590 (2.12×) |

A balanced split by FLOPs:

- PP=2: about 19/9 layers at s=512 and 18/10 at s=2048.
- PP=4: the head alone is roughly one quarter of the total, so the ideal is about 9/9/9/1. rdsp's `ExplicitCuts` requires at least one layer per stage, so use cuts (9, 18, 27).
- Megatron equivalents: `--decoder-last-pipeline-num-layers`, or a layout string such as `"Et*9|t*9|t*9|t,L"`.

This "balanced split" row is **expressible in both** frameworks. It is not an rdsp-only capability.

### 3.5 Memory

- Framework-neutral and primary: an NVML sampler thread in the Modal wrapper polls `nvmlDeviceGetMemoryInfo(...).used` on every GPU at 20 Hz. Report the peak per GPU, which includes the CUDA context and the allocator cache.
- Secondary: `torch.cuda.max_memory_allocated()` and `max_memory_reserved()` per stage. For rdsp, reset after warm-up from the engine proxy. For Megatron, read them in the custom loop; `pretrain_gpt.py`'s memory report only covers some ranks.

### 3.6 Inter-stage communication

Three views:

1. **Pure transfer micro-benchmark**, same tensor sizes (b·s·1024·2 B = 4 MiB per boundary here), on 2 GPUs:
   - (a) rdsp path: `.cpu()` → Ray object store → `.to(cuda)`, timed inside two actors (RDT.md §4.3A);
   - (b) RDT NCCL;
   - (c) `torch.distributed` `isend`/`irecv` (what Megatron uses);
   - (d) an empty Ray task, for fixed overhead.

   Report p50/p95 over ≥200 repetitions.
2. **Exposed transport lag inside the pipeline** (rdsp). For each command c on stage i: `lag(c) = start(c) − max(end(previous command on i), end(producer of c's input))`. It is the time spent after both the consumer was free and the data was produced. So it is pure transport + dispatch + deserialization. Timestamps come from the Ray timeline and from `executed_commands()` (§5.2).
3. **Megatron p2p timers** (`forward-recv`, `backward-send-forward-recv`, …). These **include waiting for the peer**, so they mix bubble and transfer. Report them as "p2p time (incl. wait)" and do not compare them directly to view 2.

### 3.7 Loss-trajectory agreement from identical weights

This is feasible, using Megatron-Bridge:

1. Save one untied HF checkpoint to the Modal volume: load Qwen3-0.6B, `lm_head.weight = clone(embed)`, `config.tie_word_embeddings = False`, `save_pretrained`. Both sides load *this* directory. Bridge maps `tie_word_embeddings` → `share_embeddings_and_output_weights` [src: `model_bridge.py:452`], so the untied checkpoint loads `lm_head.weight` into `output_layer.weight` [src: `qwen3_bridge.py` mapping].
2. Use a fixed real-text batch (tokenized once, saved as a `.pt`), so the loss visibly drops as the model overfits it, as in P6. Both scripts load the same tensor.
3. Gates:
   - (G1) step-0 **eval** loss: rdsp vs HF unsplit vs Megatron, pairwise |Δ|/loss < 1e-2;
   - (G2) 20 training steps at lr 1e-5, clipping off, weight decay 0: max per-step relative difference < 2e-2 and no monotone drift. **[unverified]** thresholds; bf16 kernel differences (TE fused norm+linear, RoPE implementation, attention kernel) set the floor.
   - Plot all three curves (rdsp, Megatron, HF single-GPU AdamW).

---

## 4. Experiment matrix

### 4.1 Systems

| ID | System | Available today? |
|---|---|---|
| R-os | rdsp, Ray object store transport (current code) | yes |
| R-rdt | rdsp, RDT NCCL transport | **no**: needs the change sketched in RDT.md §4.2 (a source change; out of scope here). The driver in §9.3 takes `--transport` and fails fast until the `create_stage_clients(transport=...)` keyword exists. |
| M-plain | Megatron, matched (§2.3) | yes |
| M-defaults | Megatron, defaults (§2.3) | yes |
| M-vpp | Megatron interleaved | yes |

### 4.2 Core grid (per system, per GPU type)

PP ∈ {2, 4} × m ∈ {4, 8, 16} × s ∈ {512, 2048}, with (b, s) = (4, 512) or (1, 2048). That is **12 cells** per system per GPU type. GPU types: **L4** (`gpu="L4:p"`) and **H100** (`gpu="H100!:p"`). Headline cells get 3 repeats in fresh containers. The rest get 1 repeat.

### 4.3 Extra rows

| Row | Systems | Cells |
|---|---|---|
| PP=1 baselines (for E) | Megatron PP=1; DeepSpeed 1-GPU; rdsp `stages=1` | s ∈ {512, 2048}, m=8, both GPU types |
| Interleaved | M-vpp | PP ∈ {2, 4} × m ∈ {8, 16} × s ∈ {512, 2048} |
| Balanced split | R-os (`ExplicitCuts`) vs M-plain (layout) | PP ∈ {2, 4}, m=16, s ∈ {512, 2048} |
| Transfer micro-benchmark | object store / RDT / NCCL p2p / empty task | 2 GPUs, 64 KiB–256 MiB sweep |
| **Capability** | §4.4 | 5 GPUs |

### 4.4 Capability row: extra GPUs on the head-heavy last stage

- rdsp: `PipelineConfig(stages=4, partition=UniformTransformerBlocks(), stage_overrides=(StageOverride(stage=3, num_gpus=2),))` on 5 GPUs. `StageOverride` lives in `ray_deepspeed_pipeline.config`; it is not re-exported from the package root.
- Megatron cannot express this. The world size must factor as TP×PP×CP×DP with one uniform shape. Its same-budget alternatives:
  - (a) **PP=5** with an uneven layout such as `"Et*7|t*7|t*7|t*6|t,L"`, on 5 GPUs;
  - (b) PP=4 + data parallel 2 = 8 GPUs;
  - (c) PP=4 on 4 GPUs, leaving one GPU idle.
- **Important caveat [src].** `StageGroupClient.submit()` sends the *same* `inputs`/`upstream` to every rank of a stage and returns rank 0's handle (`stage_worker.py`). Each rank of the 2-GPU last stage therefore computes the **same** microbatch, and DeepSpeed all-reduces identical gradients.
  - Expected result today: **no speed-up**, plus a small all-reduce cost. The row demonstrates that the configuration is expressible and correct (loss parity) and what it costs, not a benefit.
  - The throughput benefit needs a design change (a source change, not in scope): split each microbatch's rows across the stage's ranks, and scatter/gather at the stage boundary. After that change, re-run the row unchanged. The ENGINEERING.md "Honest edges" section already flags data parallelism inside a stage as not yet validated.

---

## 5. Instrumenting rdsp without changing its public API

When this plan was written, the public `rdsp.initialize()` always used the generic `build_stage_module`. That builder cannot run Qwen3, which needs per-call rotary position inputs. So, like the P6 test, the benchmark assembles the engine from the same internal pieces: `lower()` → `create_stage_clients(..., stage_builder=..., engine_factory=...)` → `PipelineCoordinator` → `RayPipelineEngine`. These are existing keyword seams. Nothing is patched.

**Update 2026-09-25 (user decision):** `create_stage_clients` now picks the stage builder from the model when none is passed (`partition.select_stage_builder`). HF llama-style causal LMs (`embed_tokens`, `norm` and `rotary_emb` next to the block list, plus a top-level `lm_head`) get `build_causal_lm_stage`; other models keep `build_stage_module`. So `rdsp.initialize()` and recovery rebuilds now work on Qwen3/Llama directly. An explicit `stage_builder=` (as the benchmark and tests pass) still takes precedence.

**Update 2026-09-26 (active phase, top of this file):** a third outcome. HF llama-style causal LMs whose parameters are *only* embedding, blocks, norm and head keep `build_causal_lm_stage`. Any other HF model (it has a `config`, e.g. Qwen3-VL with its vision encoder) gets `hf_stage.build_hf_stage`, which runs the model's own forward. Non-HF models keep `build_stage_module`. Refined the same day: `build_causal_lm_stage` is kept only for the model types it was validated on (`qwen3`, `qwen3_moe`, `llama`); every other HF model, including Llama-shaped ones such as Gemma (embedding scaling), Gemma2/Cohere (logit soft-capping) and Qwen3.5/Gemma3 (per-layer-type arguments), gets `build_hf_stage`. `tests/integration/test_hf_families.py` checks 24 families through the default routing, plus GLM-5.3 (transformers 5.17).

### 5.1 Engine proxy (`engine_factory` seam)

`create_stage_clients(engine_factory=f)` passes `f(stage_module, ds_config)` into each actor's `DeepSpeedStageAdapter` [src].

- Our `f` calls rdsp's own default factory (`deepspeed_adapter._deepspeed_engine_factory`: reused, not modified).
- It wraps the result in `TimedEngine`, a delegating proxy. The adapter only touches `engine(x)`, `.backward`, `.step`, `.module`, `.optimizer`, `.zero_grad`, `.save_checkpoint` and `.load_checkpoint` [src], and `__getattr__` forwards the rest.
- The proxy records a CUDA event pair and a host `time.monotonic_ns()` stamp around each forward, backward and step. It adds an NVTX range for each. It synchronizes **once per optimizer step**, inside `step()`, which runs after the all-stage ready barrier anyway.
- It writes one JSON line per step to a local file (`/tmp/rdsp_bench/<tag>.rank<r>.jsonl`) holding GPU-ms per op, `max_memory_allocated`/`reserved`, and the PID. It resets peak memory stats after warm-up.
- Files avoid needing any new actor method. The driver reads them after the run (same container).

### 5.2 Stage tags (`stage_builder` seam)

A wrapper around `build_causal_lm_stage` sets `module.bench_tag = f"blocks{start:02d}-{stop:02d}"`, so the proxy knows which stage it is. The driver maps tags to stage indices through `plan.stages`.

### 5.3 Host-side timeline (no code at all)

- `ray.timeline(filename=...)` after the measured steps gives a Chrome trace with every `StageWorkerActor.execute` task's start and end, per worker process [doc: Ray profiling guide].
- The actor method `executed_commands()`, which already exists, returns command ids in execution order. The i-th `execute` event on an actor is the i-th id [src: `stage_worker.py`].
- Join the two, plus proxy PIDs, to get per-command host intervals. From these:
  - adapter overhead = execute interval − GPU op time, which is mainly `.cpu()` D2H and `.to(device)` H2D;
  - transport lag (§3.6 view 2);
  - dispatch gaps.

### 5.4 Nsight Systems (diagnostic cells only)

- Ray supports `runtime_env={"nsight": {...}}` for workers [doc: Ray profiling guide]. The actors are created inside `create_stage_clients` with no per-actor `runtime_env`, so pass it job-wide via `ray.init(runtime_env=...)`. **[unverified]**: that a job-level `nsight` entry profiles every actor.
- Combined with the NVTX ranges from 5.1, this shows copy engines vs compute overlap.
- Megatron side: `nsys profile -t cuda,nvtx torchrun ...` with `--profile --profile-step-start 20 --profile-step-end 25`.
- Use nsys on 2 cells per system, not the whole grid.

### 5.5 Clean vs instrumented runs

Throughput numbers come from **clean** runs: CUDA events on (cheap, no sync), no Ray timeline dump, no nsys, Megatron `--timing-log-level 0`. Bubble and communication breakdowns come from separate **instrumented** runs: Ray timeline on, Megatron `--timing-log-level 2`. Report the instrumented runs' step-time inflation, so readers can see the probe cost.

---

## 6. Other methodology rules

- Record in every result line: image tag, `pip freeze` hash, torch/TE/NCCL/DeepSpeed/Ray/megatron-core versions, GPU name/UUID/SM clock, `nvidia-smi topo -m`, and NCCL transport lines.
- Randomize cell order within a container and alternate systems (A B A B).
- Headline claims need all 3 repeats to agree in sign. Report the median of per-run medians and the min–max across repeats. Call two systems "different" only if the min–max ranges do not overlap. With 3 repeats a formal test has little power, so say so.
- Report failures (out-of-memory, hangs) as results, not gaps. A 24 GB L4 at PP=2, s=2048, m=16 is the most likely out-of-memory cell for rdsp, because the last stage holds fp32 logits of 2048×151936 = 1.24 GB per microbatch in the loss [calc].

---

## 7. Modal specifics

**Superseded for the active phase (user decision, 2026-09-26/27).** The Qwen3-0.6B
benchmark in this plan ran on Modal as described below. The 30B vision-language
phase (top of this file) runs on the DeepSpeed-provided AWS account instead:

- Execution: one-time spot EC2 instances on the Deep Learning Base OSS Nvidia
  Driver AMI (Ubuntu 22.04), launched per run. g6.12xlarge = 4xL4, g6.48xlarge = 8xL4,
  g6e.48xlarge = 8xL40S. Stack: Python 3.12 venv, latest torch, DeepSpeed at the
  pinned `53a2ac4`, transformers 5.17.
- Completed: full GPU suite on 8xL4, 2026-09-27, 40 passed. Next: `train_vl.py`
  with Qwen3-VL-2B on 4xL4 (parity against the unsplit model, then layouts).

### 7.1 Image

- **One NGC-based image for both frameworks** (§9.4): `modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.01-py3")`. It ships Python 3.12, which rdsp needs (≥3.12).
- Set `PIP_CONSTRAINT=""`, because NGC pins packages globally [doc: Megatron install guide].
- Install into it:
  - Megatron-LM at `core_v0.19.2` (editable; `pretrain_gpt.py` is not in the wheel);
  - `megatron-bridge==0.6.2` (try `--no-deps` plus a minimal dependency list first; its full dependency list pulls in flashinfer, diffusers, mlflow and others) **[unverified]**;
  - `transformers==5.12.1`, `ray==2.58.0`;
  - the pinned DeepSpeed revision with `--no-build-isolation` and `DS_BUILD_OPS=0` (as in `scripts/modal_tests.py`);
  - `cupy-cuda13x` for RDT;
  - `nvidia-ml-py`.
- rdsp: `add_local_dir` last, then `pip install -e` at runtime, as the current harnesses do.
- If rdsp does not work on NGC's torch 2.10 (rdsp's lock has 2.13), fall back to two images: the rdsp one from `scripts/modal_tests.py`, and the NGC one for Megatron. In that case lean on the kernel-normalized efficiency E, and state the stack difference.

### 7.2 Launch

- Megatron: `torchrun --standalone --nproc_per_node p` inside one container (p ≤ 8), launched via `subprocess` from the Modal function.
- rdsp: `ray.init()` inside the same kind of container. Request `cpu=16, memory=64 GiB`: each Ray actor takes a CPU, and object-store copies need memory bandwidth.
- Keep one Modal function per GPU shape (L4:1/2/4/5 and H100!:1/2/4/5), as in `scripts/modal_tests.py`.
- Up to 8 GPUs per container are supported for L4 and H100 [doc]. 5-GPU containers are **[unverified]**; if they fail, request 8 and set `CUDA_VISIBLE_DEVICES`, which is billed for 8.

### 7.3 Time per cell

Assumes about 250 TFLOPS/GPU achieved on H100 and about 60 on L4 while busy, the speed-ups in §3.3, plus about 0.3–1.5 min of startup: model load, Ray + DeepSpeed init, or torchrun + TE init. P6's whole acceptance run took 76 s [doc: ENGINEERING.md].

| GPU | step time range (m=4…16) | cell = 10 warm-up + 50 measured + startup |
|---|---|---|
| H100 | ~0.05–0.4 s (rdsp dispatch may add tens of ms per step) | ~1.5–2 min |
| L4 | ~0.4–1.6 s | ~3–3.5 min |

### 7.4 Cost

Modal prices [doc]: L4 $0.80/h, H100 $3.95/h, CPU $0.047/core-h, memory $0.008/GiB-h. That makes about $1.26 per container-hour for 16 cores and 64 GiB.

| Block | L4 $ | H100 $ |
|---|---|---|
| Core grid, 4 systems × 12 cells × 3 repeats | 28.1 | 52.6 |
| PP=1 baselines | 0.5 | 0.8 |
| Interleaved (Megatron) | 1.6 | 2.9 |
| Balanced split | 1.6 | 2.9 |
| Capability (5 GPUs) | 1.6 | 3.4 |
| **Subtotal** | **~33** | **~63** |

- Per cell: an L4 PP=4 cell ≈ $0.22; an H100 PP=4 cell ≈ $0.45.
- Tier 0 (smoke test + parity + micro-benchmark) adds about $5–10. Allow 1.5–2× for debugging, giving **$150–200 total**.
- Dropping R-rdt until it exists removes about 25% of the core grid.
- Wall-clock: about 8 container-hours on L4 and about 4 on H100 if run serially. Running shapes in parallel brings this to about 1–2 h.

---

## 8. Order of work

**Tier 0: environment and correctness gates**, on H100!:2, about 30 min:

1. The image builds. `pretrain_gpt.py --help` has the flags. TE imports. The attention backend is found. The CUDA 13.1-on-driver-580 question is answered (fall back to 25.10 if not).
2. P6 test passes in the NGC image with transformers 5.12.1.
3. Parity G1/G2 (§3.7): rdsp vs Megatron (Bridge) vs HF, PP=2, s=512.
4. Transfer micro-benchmark (§3.6 view 1).

**Tier 1: headline**, H100 then L4. M-plain vs R-os, full core grid, 3 repeats, plus PP=1 baselines.

**Tier 2: context rows.** M-defaults, M-vpp, balanced split, capability, instrumented runs.

**Tier 3: R-rdt**, once the transport change lands. Same cells as Tier 1.

---

## 9. Scripts (ready to adapt; none has been run)

Suggested location: `rdsp/bench/` (new directory; no library code is touched).

### 9.1 `bench/megatron_pretrain.sh`

```bash
#!/usr/bin/env bash
# Megatron-LM pretrain_gpt.py, Qwen3-0.6B shape, mock data. One cell.
#   PP=2 M=8 SEQ=2048 VARIANT=plain ./megatron_pretrain.sh
#   PP=4 M=16 SEQ=512 VARIANT=vpp   ./megatron_pretrain.sh
#   VARIANT: plain | defaults | vpp | local
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
  --eval-iters 0 --log-interval 1 --log-throughput
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
# Optional balanced split (PP=4): IMPL+=( --pipeline-model-parallel-layout "Et*9|t*9|t*9|t,L" )

cd "$MEGATRON"
exec torchrun --standalone --nproc_per_node "$PP" pretrain_gpt.py \
  "${MODEL[@]}" "${TRAIN[@]}" "${PAR[@]}" "${IMPL[@]}"
# Parse from stdout: "elapsed time per iteration (ms)", "throughput per GPU",
# "lm loss", and at TIMING=2 the per-rank forward-compute/backward-compute/*-send/*-recv.
```

### 9.2 `bench/megatron_loop.py`: Megatron-Core + Bridge, identical weights and data

```python
"""torchrun --standalone --nproc_per_node P megatron_loop.py --hf /cache/qwen3-0.6b-untied \
      --pp P --m 8 --seq 2048 --steps 60 --data /cache/batches_s2048_b1_m8.pt --out /results/x.jsonl
Megatron-Core schedule driven directly; HF weights via Megatron-Bridge AutoBridge.
SKETCH: provider attribute names follow Bridge 0.6.2 GPTModelProvider; verify in Tier 0."""
import argparse, json, os, time
import torch
from megatron.bridge import AutoBridge
from megatron.core.distributed import DistributedDataParallelConfig, finalize_model_grads
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.transformer.enums import AttnBackend

p = argparse.ArgumentParser()
for k, t, d in [("hf", str, None), ("pp", int, 2), ("vpp", int, 0), ("m", int, 8), ("seq", int, 2048),
                ("steps", int, 60), ("warmup", int, 10), ("data", str, None), ("out", str, None),
                ("fp32_grads", int, 0), ("plain", int, 1)]:
    p.add_argument(f"--{k}", type=t, default=d)
a = p.parse_args()
b = 4 if a.seq == 512 else 1

torch.distributed.init_process_group("nccl")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

bridge = AutoBridge.from_hf_pretrained(a.hf, torch_dtype=torch.bfloat16)
prov = bridge.to_megatron_provider(load_weights=True)
prov.tensor_model_parallel_size = 1
prov.pipeline_model_parallel_size = a.pp
prov.virtual_pipeline_model_parallel_size = a.vpp or None
prov.pipeline_dtype = prov.params_dtype = torch.bfloat16
prov.seq_length = a.seq
prov.attention_dropout = prov.hidden_dropout = 0.0
if a.plain:  # mirror M-plain
    prov.attention_backend = AttnBackend.flash
    prov.gradient_accumulation_fusion = False
    prov.bias_activation_fusion = False
    prov.apply_rope_fusion = False
    prov.bias_dropout_fusion = False
    prov.persist_layer_norm = False
prov.finalize()
prov.initialize_model_parallel(seed=1234)
model = prov.provide_distributed_model(
    ddp_config=DistributedDataParallelConfig(grad_reduce_in_fp32=bool(a.fp32_grads),
                                             overlap_grad_reduce=False,
                                             use_distributed_optimizer=False),
    bf16=True, wrap_with_ddp=True)
opt = get_megatron_optimizer(OptimizerConfig(
    optimizer="adam", lr=1e-5, min_lr=1e-5, weight_decay=0.0, adam_beta1=0.9,
    adam_beta2=0.999, adam_eps=1e-8, clip_grad=0.0, bf16=True,
    use_distributed_optimizer=False), model)
cfg = model[0].config
cfg.finalize_model_grads_func = finalize_model_grads
cfg.no_sync_func = [c.no_sync for c in model] if len(model) > 1 else model[0].no_sync

# data: tensor [steps, m, b, s+1] shared with the rdsp driver (same file, same order)
X = torch.load(a.data)
def batches(step):
    for k in range(a.m):
        x = X[step % X.shape[0], k]
        yield x[:, :-1].contiguous(), x[:, 1:].contiguous()

pos = torch.arange(a.seq, device="cuda").unsqueeze(0).expand(b, -1)
def forward_step(it, mdl):
    tokens, labels = next(it)
    out = mdl(tokens.cuda(non_blocking=True), pos, None, labels=labels.cuda(non_blocking=True))
    def loss_func(o):               # o = per-token CE [b, s] on the last stage
        loss = o.float().mean()
        return loss, {"lm loss": loss.detach()}
    return out, loss_func

fb = get_forward_backward_func()
last = torch.distributed.get_rank() == torch.distributed.get_world_size() - 1
times, losses = [], []
for step in range(a.steps):
    for c in model:
        c.zero_grad_buffer()
    opt.zero_grad()
    it = batches(step)
    its = [it] * len(model) if len(model) > 1 else it   # VPP: one iterator per chunk
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = fb(forward_step_func=forward_step, data_iterator=its,
             model=model if len(model) > 1 else model[0],
             num_microbatches=a.m, seq_length=a.seq, micro_batch_size=b, forward_only=False)
    opt.step()
    torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
    if step == a.warmup - 1:
        torch.cuda.reset_peak_memory_stats()
    if last:
        losses.append(sum(float(d["lm loss"]) for d in out) / a.m)

rec = {"system": "megatron_core", "pp": a.pp, "vpp": a.vpp, "m": a.m, "seq": a.seq, "b": b,
       "rank": torch.distributed.get_rank(), "step_s": times[a.warmup:],
       "max_alloc": torch.cuda.max_memory_allocated(),
       "max_reserved": torch.cuda.max_memory_reserved(), "loss": losses}
with open(f"{a.out}.rank{rec['rank']}", "w") as f:
    json.dump(rec, f)
```

**Fallback if Bridge will not install.** A direct HF→Megatron state-dict mapping for Qwen3 is about 40 lines:

- concatenate per-KV-group Q/K/V rows into `linear_qkv.weight`, following `QKVMapping`;
- concatenate gate and up into `linear_fc1.weight`;
- rename the norms to TE's fused names (table in `qwen3_bridge.py`).

### 9.3 `bench/rdsp_bench.py` and `bench/bench_hooks.py`

```python
# bench/bench_hooks.py  -- importable inside actors (PYTHONPATH via runtime_env)
import json, os, time
import torch
from ray_deepspeed_pipeline.deepspeed_adapter import _deepspeed_engine_factory  # reused, unmodified
from ray_deepspeed_pipeline.partition import build_causal_lm_stage

OUT = os.environ.get("RDSP_BENCH_DIR", "/tmp/rdsp_bench")
WARMUP = int(os.environ.get("RDSP_BENCH_WARMUP", "10"))

def tagged_stage_builder(model, start, stop, names):
    m = build_causal_lm_stage(model, start, stop, names)
    m.bench_tag = f"blocks{start:02d}-{stop:02d}"
    return m

class TimedEngine:
    """Delegating proxy: CUDA-event timing + NVTX per forward/backward/step;
    one host sync per optimizer step; one JSON line per step."""
    def __init__(self, engine, tag):
        self._e, self._tag, self._ops, self._step = engine, tag, [], 0
        os.makedirs(OUT, exist_ok=True)
        self._f = open(f"{OUT}/{tag}.rank{os.environ.get('RANK', '0')}.jsonl", "a")

    def __getattr__(self, name):          # module, optimizer, zero_grad, save/load_checkpoint
        return getattr(self._e, name)

    def _timed(self, kind, fn, *args):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t = time.monotonic_ns()
        torch.cuda.nvtx.range_push(f"{self._tag}.{kind}")
        s.record(); out = fn(*args); e.record()
        torch.cuda.nvtx.range_pop()
        self._ops.append((kind, t, s, e))
        return out

    def __call__(self, x):
        return self._timed("fwd", self._e, x)

    def backward(self, loss):
        return self._timed("bwd", self._e.backward, loss)

    def step(self):
        out = self._timed("opt", self._e.step)
        torch.cuda.synchronize()
        self._f.write(json.dumps({
            "tag": self._tag, "pid": os.getpid(), "step": self._step,
            "ops": [(k, t, s.elapsed_time(e)) for k, t, s, e in self._ops],
            "max_alloc": torch.cuda.max_memory_allocated(),
            "max_reserved": torch.cuda.max_memory_reserved()}) + "\n")
        self._f.flush()
        self._ops.clear(); self._step += 1
        if self._step == WARMUP:
            torch.cuda.reset_peak_memory_stats()
        return out

def timed_engine_factory(stage_module, ds_config):
    return TimedEngine(_deepspeed_engine_factory(stage_module, ds_config),
                       getattr(stage_module, "bench_tag", "stage"))
```

```python
# bench/rdsp_bench.py -- one cell. Same internal assembly as tests/integration/test_p6_first_row.py.
import argparse, json, os, time
import torch, torch.nn.functional as F
import ray, transformers
import ray_deepspeed_pipeline as rdsp
from ray_deepspeed_pipeline.compiler import lower
from ray_deepspeed_pipeline.config import StageOverride
from ray_deepspeed_pipeline.coordinator import PipelineCoordinator
from ray_deepspeed_pipeline.engine import RayPipelineEngine
from ray_deepspeed_pipeline.stage_worker import create_stage_clients
from bench_hooks import tagged_stage_builder, timed_engine_factory

def loss_fn(out, labels):            # labels already shifted by the data file
    return F.cross_entropy(out.float().reshape(-1, out.shape[-1]), labels.reshape(-1))

p = argparse.ArgumentParser()
for k, t, d in [("hf", str, "/cache/qwen3-0.6b-untied"), ("pp", int, 2), ("m", int, 8),
                ("seq", int, 2048), ("steps", int, 60), ("warmup", int, 10),
                ("data", str, None), ("out", str, None), ("cuts", str, ""),
                ("last_stage_gpus", int, 1), ("transport", str, "object_store"),
                ("timeline", int, 0)]:
    p.add_argument(f"--{k}", type=t, default=d)
a = p.parse_args()
b = 4 if a.seq == 512 else 1
os.environ["RDSP_BENCH_WARMUP"] = str(a.warmup)

ray.init(include_dashboard=False, log_to_driver=False,
         runtime_env={"env_vars": {"PYTHONPATH": "/root/bench",
                                   "RDSP_BENCH_WARMUP": str(a.warmup)}})
model = transformers.AutoModelForCausalLM.from_pretrained(a.hf, dtype=torch.bfloat16)
assert model.lm_head.weight is not model.model.embed_tokens.weight   # untied checkpoint

ds = {"train_micro_batch_size_per_gpu": b, "gradient_accumulation_steps": a.m,
      "train_batch_size": b * a.m,
      "optimizer": {"type": "AdamW", "params": {"lr": 1e-5, "betas": [0.9, 0.999],
                    "eps": 1e-8, "weight_decay": 0.0, "torch_adam": True, "fused": True}},
      "zero_optimization": {"stage": 0}, "bf16": {"enabled": True},
      "gradient_clipping": 0.0, "steps_per_print": 10**9}
part = (rdsp.ExplicitCuts(tuple(int(c) for c in a.cuts.split(","))) if a.cuts
        else rdsp.UniformTransformerBlocks())
overrides = ((StageOverride(stage=a.pp - 1, num_gpus=a.last_stage_gpus),)
             if a.last_stage_gpus > 1 else ())
plan = lower(model, rdsp.PipelineConfig(stages=a.pp, partition=part, microbatches=a.m,
                                        stage_overrides=overrides), ds)
kw = {} if a.transport == "object_store" else {"transport": a.transport}  # RDT: needs RDT.md §4.2
clients = create_stage_clients(model, plan, loss_fn, use_gpu=True,
                               stage_builder=tagged_stage_builder,
                               engine_factory=timed_engine_factory, **kw)
engine = RayPipelineEngine(PipelineCoordinator(plan, clients))
del model                                                            # driver holds no weights

X = torch.load(a.data)                                               # [steps, m, b, s+1]
def mbs(step):
    return iter([(X[step % X.shape[0], k][:, :-1].contiguous(),
                  X[step % X.shape[0], k][:, 1:].contiguous()) for k in range(a.m)])

times, losses = [], []
for step in range(a.steps):
    t0 = time.perf_counter()
    losses.append(float(engine.train_batch(data_iter=mbs(step))))
    times.append(time.perf_counter() - t0)

if a.timeline:
    ray.timeline(filename=a.out + ".timeline.json")
order = {c.stage: [ray.get(act.executed_commands.remote()) for act in c.actors] for c in clients}
json.dump({"system": "rdsp", "transport": a.transport, "pp": a.pp, "m": a.m, "seq": a.seq,
           "b": b, "cuts": a.cuts, "last_stage_gpus": a.last_stage_gpus,
           "stage_tags": [f"blocks{s.block_start:02d}-{s.block_stop:02d}" for s in plan.stages],
           "step_s": times[a.warmup:], "loss": losses, "executed": order},
          open(a.out, "w"))
ray.shutdown()
# Post-processing (separate script): read /tmp/rdsp_bench/*.jsonl for per-stage GPU busy
# time and memory; join timeline 'execute' events with `executed` ids for lag (§3.6).
```

Shared data file, written once per (s, b, m) by a tiny helper so both frameworks read identical ids:

```python
g = torch.Generator().manual_seed(0)
torch.save(torch.randint(0, 151936, (8, m, b, s + 1), generator=g), f"/cache/batches_s{s}_b{b}_m{m}.pt")
```

For parity (§3.7), use a real-text tensor of the same shape instead.

### 9.4 `bench/modal_bench.py`: Modal app

```python
"""modal run bench/modal_bench.py --gpu H100 --tier 1
One NGC image for both frameworks; one function per GPU shape; results to a Volume."""
import itertools, json, random
import modal

NGC = "nvcr.io/nvidia/pytorch:26.01-py3"      # fallback: 25.10-py3 (CUDA 13.0.2) if driver 580 rejects 13.1
MCORE_TAG = "core_v0.19.2"
BRIDGE = "0.6.2"
DS_REV = "53a2ac44fb664bea838df3981ba4366b91643070"

image = (
    modal.Image.from_registry(NGC)
    .env({"PIP_CONSTRAINT": "", "DS_BUILD_OPS": "0", "TOKENIZERS_PARALLELISM": "false",
          "HF_HOME": "/cache/hf", "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT"})
    .run_commands(
        f"git clone --depth 1 --branch {MCORE_TAG} https://github.com/NVIDIA/Megatron-LM.git /opt/Megatron-LM",
        "pip install --no-build-isolation -e '/opt/Megatron-LM[training]'",
        f"pip install --no-deps megatron-bridge=={BRIDGE}",
        "pip install 'transformers==5.12.1' accelerate safetensors omegaconf rich einops "
        "'ray[default]==2.58.0' cloudpickle nvidia-ml-py cupy-cuda13x",
        f"pip install --no-build-isolation git+https://github.com/deepspeedai/DeepSpeed.git@{DS_REV}",
    )
    .add_local_dir("rdsp", "/root/rdsp", ignore=[".venv", "**/__pycache__", ".pytest_cache", "uv.lock"])
    .add_local_dir("rdsp/bench", "/root/bench")
)
app = modal.App("rdsp-vs-megatron", image=image)
cache = modal.Volume.from_name("rdsp-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("rdsp-bench-results", create_if_missing=True)
KW = dict(timeout=4 * 3600, cpu=16, memory=65536,
          volumes={"/cache": cache, "/results": results})

def _env_report():
    import subprocess
    run = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout
    return {"topo": run("nvidia-smi topo -m"),
            "gpus": run("nvidia-smi --query-gpu=name,uuid,clocks.max.sm --format=csv"),
            "freeze": run("pip freeze | sha256sum")}

def _nvml_peak(stop, peaks):
    import time, pynvml
    pynvml.nvmlInit()
    hs = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    while not stop.is_set():
        for i, h in enumerate(hs):
            peaks[i] = max(peaks.get(i, 0), pynvml.nvmlDeviceGetMemoryInfo(h).used)
        time.sleep(0.05)

def run_cells(cells):
    import subprocess, threading, os, uuid
    subprocess.run(["pip", "install", "-e", "/root/rdsp"], check=True, capture_output=True)
    env = _env_report()
    random.shuffle(cells)                                  # order randomized within container
    for c in cells:
        stop, peaks = threading.Event(), {}
        th = threading.Thread(target=_nvml_peak, args=(stop, peaks), daemon=True); th.start()
        out = f"/results/{c['system']}_{c['gpu']}_pp{c['pp']}_m{c['m']}_s{c['seq']}_{uuid.uuid4().hex[:6]}"
        if c["system"].startswith("M-") and c.get("launcher") == "pretrain":
            cmd = (f"PP={c['pp']} M={c['m']} SEQ={c['seq']} VARIANT={c['variant']} "
                   f"TIMING={c.get('timing', 0)} bash /root/bench/megatron_pretrain.sh")
        elif c["system"].startswith("M-"):
            cmd = (f"cd /root/bench && torchrun --standalone --nproc_per_node {c['pp']} megatron_loop.py "
                   f"--hf /cache/qwen3-0.6b-untied --pp {c['pp']} --m {c['m']} --seq {c['seq']} "
                   f"--data {c['data']} --out {out}")
        else:
            cmd = (f"cd /root/bench && python rdsp_bench.py --pp {c['pp']} --m {c['m']} --seq {c['seq']} "
                   f"--data {c['data']} --out {out}.json --transport {c.get('transport', 'object_store')} "
                   f"--cuts '{c.get('cuts', '')}' --last_stage_gpus {c.get('last_stage_gpus', 1)}")
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=os.environ)
        stop.set(); th.join()
        with open(out + ".meta.json", "w") as f:
            json.dump({"cell": c, "rc": r.returncode, "nvml_peak_bytes": peaks, "env": env,
                       "stdout_tail": r.stdout[-20000:], "stderr_tail": r.stderr[-8000:]}, f)
        results.commit()
    return len(cells)

@app.function(gpu="L4:2", **KW)
def l4x2(cells): return run_cells(cells)
@app.function(gpu="L4:4", **KW)
def l4x4(cells): return run_cells(cells)
@app.function(gpu="H100!:2", **KW)
def h100x2(cells): return run_cells(cells)
@app.function(gpu="H100!:4", **KW)
def h100x4(cells): return run_cells(cells)
# add :1 (baselines) and :5 (capability; [unverified] that 5 is allowed, else :8 + CUDA_VISIBLE_DEVICES)

@app.function(**KW)   # CPU-only: untied checkpoint + data files, once
def prepare():
    import torch, transformers
    m = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
    m.lm_head.weight = torch.nn.Parameter(m.model.embed_tokens.weight.detach().clone())
    m.config.tie_word_embeddings = False
    m.save_pretrained("/cache/qwen3-0.6b-untied")
    transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B").save_pretrained("/cache/qwen3-0.6b-untied")
    for s, m_ in itertools.product((512, 2048), (4, 8, 16)):
        b = 4 if s == 512 else 1
        g = torch.Generator().manual_seed(0)
        torch.save(torch.randint(0, 151936, (8, m_, b, s + 1), generator=g),
                   f"/cache/batches_s{s}_b{b}_m{m_}.pt")
    cache.commit()

RUN = {("L4", 2): l4x2, ("L4", 4): l4x4, ("H100", 2): h100x2, ("H100", 4): h100x4}

@app.local_entrypoint()
def main(gpu: str = "H100", reps: int = 3, systems: str = "M-plain,R-os"):
    prepare.remote()
    jobs = []
    for pp in (2, 4):
        cells = []
        for sysname, m, s in itertools.product(systems.split(","), (4, 8, 16), (512, 2048)):
            b = 4 if s == 512 else 1
            cells.append({"system": sysname, "gpu": gpu, "pp": pp, "m": m, "seq": s,
                          "variant": {"M-plain": "plain", "M-defaults": "defaults"}.get(sysname),
                          "launcher": "pretrain" if sysname == "M-defaults" else "loop",
                          "data": f"/cache/batches_s{s}_b{b}_m{m}.pt"})
        for _ in range(reps):                              # each repeat = fresh container
            jobs.append(RUN[(gpu, pp)].spawn(list(cells)))
    print(sum(j.get() for j in jobs), "cells done")
```

Notes on the Modal sketch:

- `add_local_dir` must stay the last image step [doc: Modal image guide]. The two harnesses in this repo follow that already.
- `gpu="H100!:2"` combines the "no upgrade" marker with a count. **[unverified]** syntax; confirm in Tier 0.
- NGC images put Python at `/usr/bin/python`. `from_registry` without `add_python` should work **[unverified]**.

---

## 10. Open items (verify in Tier 0)

1. Whether CUDA 13.1 (NGC 26.01) runs on Modal's driver 580 (CUDA 13.0). If not, use 25.10.
2. Whether TE's `flash` attention backend is available in the chosen NGC image, or `fused` must be used instead.
3. Whether the Megatron-Bridge 0.6.2 provider attribute names and `provide_distributed_model(...)` keywords match the §9.2 sketch; whether `--no-deps` install is enough.
4. Whether the rdsp P6 test passes on NGC torch 2.10 with transformers 5.12.1.
5. Whether DeepSpeed accepts `torch_adam + fused=True` under the bf16 `FP16_Optimizer` wrapper.
6. Whether `BF16_Optimizer` (ZeRO-1, fp32 accumulation) is correct under rdsp's one-step-many-backwards pattern. Only needed for the optional fp32 check.
7. Whether the PP=4 VPP layout string validates.
8. Whether 5-GPU Modal containers are allowed.
9. Whether a job-level `nsight` runtime_env covers actors created without their own runtime_env.
10. L4 and H100 dense bf16 peak values: confirm against the datasheets before quoting MFU.

---

## 11. Sources

Code (read locally in shallow clones):

- Megatron-LM `core_v0.19.2` (4b4acac): https://github.com/NVIDIA/Megatron-LM/tree/core_v0.19.2
  - `megatron/training/arguments.py`: flags; validation at L435, L893–945, L1180–1193, L2750–2810, L2942–2997
  - `megatron/training/argument_utils.py`: flags generated from dataclass fields
  - `megatron/training/training.py`: `num_floating_point_operations`, timers L2630–2652
  - `megatron/core/transformer/transformer_config.py`: `qk_layernorm`, `transformer_impl`, `layernorm_epsilon` → `--norm-epsilon`
  - `gpt_builders.py`, `megatron/core/models/gpt/gpt_layer_specs.py`, `megatron/core/models/backends.py`: local vs TE specs
  - `megatron/core/optimizer/__init__.py`: Adam class selection
  - `megatron/core/pipeline_parallel/schedules.py`, `p2p_communication.py`: 1F1B schedules and p2p timers
  - `megatron/core/transformer/pipeline_parallel_layer_layout.py`: layout string syntax
  - `megatron/core/tokenizers/text/libraries/null_tokenizer.py`
  - `docs/get-started/install.md`, `examples/llama/train_llama3_8b_h100_fp8.sh`, `examples/run_simple_mcore_train_loop.py`, `docker/Dockerfile.ci.*`
- Megatron-Bridge `v0.6.2` (c0e164e): https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/v0.6.2
  - `src/megatron/bridge/models/qwen/qwen3_bridge.py`
  - `src/megatron/bridge/models/conversion/model_bridge.py`
  - `src/megatron/bridge/models/model_provider.py`
  - `src/megatron/bridge/recipes/qwen/h100/qwen3.py`
  - `examples/conversion/hf_megatron_roundtrip_multi_gpu.py`, `examples/conversion/README.md`
  - `scripts/conversion/README.md`, `scripts/training/README.md`
  - `pyproject.toml`, `docker/Dockerfile.ci`
- DeepSpeed pinned revision 53a2ac4 (local `DeepSpeed/`):
  - `deepspeed/runtime/engine.py` L1850–1895 (optimizer wrapper choice), L1966–2010 (Adam construction), L2208 (BF16_Optimizer)
  - `deepspeed/runtime/constants.py` L247–254 (clipping default 1.0)
- rdsp (this repo): `src/ray_deepspeed_pipeline/{stage_worker,deepspeed_adapter,coordinator,schedule,partition,compiler,config}.py`, `tests/integration/test_p6_first_row.py`, `docs/ENGINEERING.md`, `docs/research/RDT.md`

Documentation:

- PyPI: https://pypi.org/project/megatron-core/ (0.19.2), https://pypi.org/project/megatron-bridge/ (0.6.2)
- Megatron-Core install guide: https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/install.html
- Megatron-Bridge README: https://github.com/NVIDIA-NeMo/Megatron-Bridge
- NGC PyTorch release notes:
  - 26.01: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-01.html
  - 25.10: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-10.html
  - 26.06: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-06.html
- Qwen3-0.6B config: https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json
- Modal:
  - CUDA/driver: https://modal.com/docs/guide/cuda
  - GPUs: https://modal.com/docs/guide/gpu
  - Pricing: https://modal.com/pricing
- Ray profiling (timeline, Nsight): https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html
- Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM", SC'21, https://arxiv.org/abs/2104.04473 (1F1B and interleaved bubble analysis)
- Chowdhery et al., "PaLM", https://arxiv.org/abs/2204.02311, Appendix B (MFU definition)
- NVIDIA datasheets (peak bf16): https://www.nvidia.com/en-us/data-center/l4/, https://www.nvidia.com/en-us/data-center/h100/
