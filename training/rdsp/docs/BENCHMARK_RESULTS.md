# Benchmark results

rdsp against MegatronMIMO and Megatron-Bridge, training Qwen3.5-2B and 4B on CORD-v2. The first set ran on one AWS g7.48xlarge: 8× RTX PRO 4500 Blackwell, 32 GB, PCIe, no NVLink. A 4B run on 8× H100 with NVLink follows. How to rerun: [GETTING_STARTED.md](GETTING_STARTED.md#reproducing-the-qwen35-2b-and-4b-results). `python bench/common/summarize.py <log directory>` prints the table rows from a run's logs. The Qwen3.5-9B, 27B and 35B-A3B results are in the [README](../README.md) and [GETTING_STARTED.md](GETTING_STARTED.md#reproducing-the-blogs-results).

## Results: 8× RTX PRO 4500, PCIe

Median over steps 6–50 of one 50-step run, 64 rows of up to 2,048 tokens a step. Cost assumes $5.32/h for 8 GPUs, prorated to 5 for the MIMO layout.

| Model | Layout | Megatron | rdsp | Step time | Megatron tok/s | rdsp tok/s | Megatron tok/s per GPU | rdsp tok/s per GPU | Megatron $/M tok | rdsp $/M tok |
|---|---|---|---|---|---|---|---|---|---|---|
| 2B | Vision 1 GPU + language TP2×DP2, vs MIMO | 9.64 s | 8.30 s | −14% | 13,595 | 15,796 | 2,719 | 3,159 | 0.068 | 0.058 |
| 2B | TP2×PP2×DP2, vs Bridge | 7.39 s | 5.30 s | −28% | 17,740 | 24,745 | 2,218 | 3,093 | 0.083 | 0.060 |
| 4B | TP2×PP2×DP2, vs Bridge | 10.04 s | 8.69 s | −13% | 13,058 | 15,087 | 1,632 | 1,886 | 0.113 | 0.098 |
| 4B | TP4×PP2×DP1, vs Bridge | 15.93 s | 13.29 s | −17% | 8,229 | 9,866 | 1,029 | 1,233 | 0.180 | 0.150 |

Loss: every pair starts from the same step-0 loss. Over steps 1–49, rdsp's mean difference from Megatron is +0.0005 (2B MIMO layout), +0.003 (4B TP2×PP2×DP2) and −0.0024 (4B TP4×PP2×DP1). Steps 4–5 vary by up to 0.08 between repeated runs of either system, from GPU kernels that aren't bit-for-bit deterministic.

The 4B Megatron numbers come from a second machine of the same type.

## Results: 8× H100, NVLink

Qwen3.5-4B on one AWS p5.48xlarge: 8× H100 SXM 80 GB, every pair of GPUs linked by NVLink through NVSwitch (`nvidia-smi topo -m`: NV18). Same software, data, settings and recipes as above. Cost assumes $22.28/h for 8 GPUs (spot, us-east-2).

| Layout | Megatron | rdsp | Step time | Megatron tok/s | rdsp tok/s | Megatron tok/s per GPU | rdsp tok/s per GPU | Megatron $/M tok | rdsp $/M tok |
|---|---|---|---|---|---|---|---|---|---|
| Vision 1 GPU + language TP2×DP2, vs MIMO | 7.99 s | 7.89 s | −1% | 16,413 | 16,606 | 3,283 | 3,321 | 0.236 | 0.233 |
| TP2×PP2×DP2, vs Bridge | 8.79 s | 6.49 s | −26% | 14,917 | 20,208 | 1,865 | 2,526 | 0.415 | 0.306 |
| TP4×PP2×DP1, vs Bridge | 18.24 s | 12.30 s | −33% | 7,184 | 10,658 | 898 | 1,332 | 0.861 | 0.581 |
| Colocated vision + language TP2×PP2×DP2 (rdsp only) | — | 5.62 s | | | 23,322 | — | 2,915 | | 0.265 |

Loss: every pair starts from the same step-0 loss; over steps 1–49 rdsp's mean difference from Megatron is +0.003, −0.003 and −0.008, and colocated's from Megatron-Bridge TP2×PP2×DP2 is −0.002.

Per GPU, MIMO's 5-GPU layout is the most efficient here (3,283 and 3,321 tokens/s per GPU); colocated is the fastest per step and the second most efficient per GPU (2,915).

**Read with care:**
- **One round.** No repeats, so no spread.
- **The host CPU limits launch-heavy runs.** This instance has an AMD EPYC 7R13. A torch-profiler step on rdsp's language GPUs in the MIMO layout shows the CPU busy for the whole step while GPU kernels run for only part of it: the layout is bound by kernel launches, not by the GPUs. That's why it ends in a near-tie. Megatron's TP4×PP2×DP1 (64 one-row microbatches per GPU, 26.7 TFLOP/s per GPU) is slower here than on the PCIe GPUs, most likely for the same reason. That's an inference, since its profiler trace was lost, and it inflates the −33%.
- **rdsp's TP4 split is off on H100.** Stage 0 (vision encoder + 12 layers) is busy 12.1 s per step while stage 1 waits 3.2 s. The balanced estimate doesn't model how much less efficiently the encoder's small matrices run under TP4.
- **Colocated ran without compiling the encoder.** The recipe's `compile_vision` doesn't apply to a colocated encoder (now rejected at start-up); `recipes/qwen35-4b-coloc-tp2pp2dp2.sh` compiles it, and is unmeasured.
- **NVLink was used by both systems.** With `NCCL_DEBUG=INFO`, every NCCL connection on both sides is `P2P/CUMEM`, with no `SHM` or `NET` fallback.
- **Defaults on both sides.** Megatron's TP communication overlap (`tp_comm_overlap`) is off in its stock recipe, as on PCIe; rdsp has no equivalent.

## rdsp settings per layout

Every run uses its recipe in `recipes/`, with `--prefetch --sharded-loss --drop-padding-mask --pad-per-microbatch` and `--cuts balanced` unless noted.

| Layout | Recipe | Extra settings | Cut chosen |
|---|---|---|---|
| Vision 1 GPU + TP2×DP2 | `qwen35-2b-vision1-tp2dp2.sh` | `--cuts 0` (vision-only stage), `--loss microbatch-mean`, 2 rows × 32 microbatches, stage 0 compiled | encoder alone |
| TP2×PP2×DP2 (2B) | `qwen35-2b-tp2pp2dp2.sh` | `--untie-embeddings`, 2 rows × 32 microbatches, vision encoder compiled | 5 of 24 layers on stage 0 |
| TP2×PP2×DP2 (4B) | `qwen35-4b-tp2pp2dp2.sh` | same | 13 of 32 |
| TP4×PP2×DP1 (4B) | `qwen35-4b-tp4pp2dp1.sh` | `--untie-embeddings`, 1 row × 64 microbatches, vision encoder compiled | 12 of 32 |

Megatron used its default even split (12 of 24, 16 of 32). It accepts uneven splits through flags; we didn't tune them.

Settings held equal: sequence length 2,048, 64 rows a step, one row per GPU per microbatch; AdamW, learning rate 1e-5 constant, betas 0.9 / 0.95, eps 1e-8, no weight decay, clipping or warm-up; bf16 compute with fp32 gradients and optimizer state split over data parallel; no recompute. `bench/common/runs.sh` turns off Megatron's defaults that differ: clipping, warm-up, cosine decay, the multi-token-prediction head and the shuffling sampler.

Differences that remain:

- **Padding.** Megatron-Bridge pads every row to 2,048 whenever it pipelines (`dataset.pad_to_max_length`, forced when PP > 1). rdsp pads each microbatch to its longest row. So in the vs-Bridge rows Bridge computes about 2 positions per real token against rdsp's 1.1–1.2, and the two do not do equal work. MegatronMIMO's padding changes its step time little (9.60 s with dynamic padding against 9.64 s).
- **Loss.** Megatron-Bridge trains on the token mean. MegatronMIMO trains on per-microbatch means and can't be changed, so the rdsp run against it does the same (`--loss microbatch-mean`).
- **Embeddings.** In the pipeline layouts Megatron ties the input embedding and the output layer across stages; rdsp trains two copies (`--untie-embeddings`).

## The balanced cut

`--cuts balanced` estimates each stage's work before training ([ENGINEERING.md](ENGINEERING.md#balanced-cuts)). We checked it by timing every candidate cut for a few steps:

| Layout | Even split | Balanced | Fastest measured |
|---|---|---|---|
| 2B TP2×PP2×DP2 | 12 | 5 (5.30 s) | 5 |
| 4B TP2×PP2×DP2 | 16 (out of memory) | 13 (8.69 s) | 13 |
| 4B TP4×PP2×DP1 | 16 (14.54 s) | 12 (13.29 s) | 11 (13.15 s) |

Under TP4 the estimate misses by one layer (about 1%): the encoder's small matrices lose efficiency when split four ways, and the estimate doesn't model that.

## What moved the numbers

rdsp started 40% slower than MegatronMIMO in MIMO's own layout. 2B step time after each change:

| Change | Step time |
|---|---|
| Start: encoder shares the first stage, with recompute | 13.51 s |
| Vision-only first stage: frees enough memory to drop recompute | 11.3 s |
| Prefetch: the next step's data (about 576 MB of images) ships while this step runs; dispatch fell from 1.1 s to 1 ms | 10.3 s (12-step run) |
| `torch.compile` on the vision encoder: its forward and backward fell from 7.8 s to 6.7 s per step | 9.62 s |
| Split-vocabulary loss: TP ranks keep their slice of the 248k-wide logits instead of gathering 2 GB per microbatch | 9.25 s |
| No padding mask: rows pad only on the right, so causal attention ignores the padding anyway, and SDPA picks its flash kernel | 9.05 s |
| bf16 gradients freed after fp32 accumulation (2 bytes per parameter less, as in Megatron), and padding per microbatch (1.71 → 1.20 padded tokens per real token) | 8.30 s |

Same chain for 4B TP2×PP2×DP2: 13.51 s → 11.57 s → 10.58 s → 8.71 s → 8.69 s. The freed gradients removed the out-of-memory errors that had forced recompute, and the balanced cut did the rest.

## Not included

- **4B in MIMO's layout.** Neither system fits it in 32 GB without recompute.
- **2B TP4.** Qwen3.5-2B has 2 key/value heads, which can't be split 4 ways.
- **9B and up.** These need 80 GB-class GPUs in these layouts.
- **Colocated vision.** rdsp only; neither Megatron system supports it for Qwen3.5. 4B ran at 9.66 s, slower than TP2×PP2×DP2; 2B didn't fit.
- **Repeats.** One 50-step run per configuration.

## Validation provenance

The supported layouts in `src/ray_deepspeed_pipeline/support_matrix.py` passed on 2 to 8 L4 GPUs (real DeepSpeed and NCCL, fp32 tiny models, SGD with momentum), with:
- loss parity against the unsplit model within 1e-4;
- per-parameter update parity within 2.4e-5 (3e-4 with AutoEP folding);
- checkpoint round trip;
- rank-kill recovery.

The two-stage baseline uses Qwen3-0.6B in bf16. The three vision placements match the unsplit model on 4× L40S (`tests/integration/test_vl_layouts_gpu.py`, tiny Qwen3.5-VL, fp32). Evidence per row is in `src/ray_deepspeed_pipeline/support_matrix.py`.
