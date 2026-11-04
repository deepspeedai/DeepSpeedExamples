# SFT eval: gc, offload, tiling, tiling_offload

ArcticTraining launches DeepSpeed. Sequence parallel stays at 1. All four arms share Liger, FlashAttention 2 or 3, fused CE, and tiled MLP. The arms differ only in rematerialize tiling and activation offload.

Source: ArcticTraining `projects/sequence-parallelism`. Capacity is the longest length that finishes. Sequence parallel is data-parallel only (`SP = 1`).

## Shared baseline

Same Arctic SFT YAML for every arm. UltraChat `train_sft`, ZeRO-2 with GPU Adam, `expandable_segments`. Same data order, `max_length`, micro-batch, gradient accumulation, and seed. ZeRO-2 GPU Adam is the floor: parameters plus Adam state on each GPU. It does not grow with sequence length or micro-batch.

| Baseline optim | Arctic setting | Notes |
| --- | --- | --- |
| Liger | `model.type: liger` | Fused kernels on the HF model. |
| FlashAttention 2 or 3 | `attn_implementation: flash_attention_3` | FA3 on H200. FA2 if FA3 is not installed. |
| Fused CE | Liger fused linear cross-entropy | Comes with `model.type: liger`. Do not materialize full logits. |
| Tiled MLP | `tiled_mlp_compute: true` | Sequence-tiled MLP on every arm. Not the difference under test. |

| Setting | Value | Why |
| --- | --- | --- |
| `sequence_parallel_size` | 1 | No Ulysses. Per-rank sequence is the full sample. |
| `zero_optimization.stage` | 2 | ZeRO-2. Optimizer is GPU Adam, not CPU Adam. |
| `offload_optimizer` / `offload_param` | off | The floor stays on GPU. Do not move Adam or weights to CPU. |
| `activation_checkpoint_cpu_offload` | off | Arctic's flag is not our offload arm. Use the DeepSpeed offload context. |
| `exit_iteration` | 4 after 1 warmup | Enough to prove the step runs. Not a quality run. |
| `comms_logger` | off | It serializes collectives and biases tokens/s. |

## Four arms

- **gc** (baseline). Shared baseline plus Hugging Face gradient checkpointing. Qwen3.5 must use reentrant checkpointing. No tiled rematerialize. No activation offload.
- **offload.** gc plus DeepSpeed activation offload of the checkpoint inputs. No tiled rematerialize. The offload context stays open across forward and backward.
- **tiling.** Shared baseline plus tiled rematerialize. Saved inputs stay on GPU. No activation offload.
- **tiling_offload.** Tiled rematerialize and activation offload together. This is the arm that finished 64k on 35B-A3B when gc did not.

## Three workloads

Same four arms on each. Sequence parallel stays 1. Qwen3.5, 8×H200. Stop an arm at its first out-of-memory. Loss must match gc on the shortest cell of that workload before the ladder counts.

### 1. Packed long sequence

Micro-batch 1, gas 1. One row is many whole samples end to end, with `cu_seqlens`. Do not cut inside a sample. Tile budget is 8192 tokens. Sweep the packed length, not the sample length.

| Packed length | What it is | How to read the arms |
| --- | --- | --- |
| 8k | One 8k sample | Tiling is a no-op. All four match gc. |
| 16k, 32k, 64k, 128k | 2, 4, 8, 16 samples of 8k | gc and offload rematerialize the whole pack. Tiling keeps the mixer at one 8k tile. |
| Past 128k | More 8k samples | Stop each arm on OOM. tiling_offload is the expected long arm. |

### 2. Non-packed long sequence

Micro-batch 1, gas 1. One sample whose length is the sequence. No `cu_seqlens` split. Same length ladder: 8k, 16k, 32k, 64k, 128k. Tiling has no boundary, so tiling matches gc and tiling_offload matches offload. This workload measures offload alone.

### 3. 8k sequence, micro-batch scaling

Each sample is 8192 and is not packed. Gas stays 1. Sweep micro-batch 1, 2, 4, 8, 16. Remat shards stay 2, so the tile grows with the micro-batch. At micro-batch 1 tiling is a no-op. From 2 up, batch-axis tiling is on. Report the largest micro-batch each arm finishes, plus GiB.

| mbs | Tokens / rank | Tiling |
| --- | --- | --- |
| 1 | 8k | No-op. tiling matches gc. tiling_offload matches offload. |
| 2 | 16k | Two shards of 1 row. First real tiling cell. |
| 4 | 32k | Two shards of 2 rows. |
| 8 | 64k | Two shards of 4 rows. |
| 16 | 128k | Two shards of 8 rows. Prior 35B gc did not fit mbs 16. |

## Report

- Workloads 1 and 2: longest packed or single-sample length.
- Workload 3: largest mbs.
- CUDA GiB.
- Loss vs gc on the shared cell.

On workload 1, read offload, tiling, and tiling_offload against gc. On workload 2, expect only two distinct results. On workload 3, the gap should open once mbs is greater than 1. Tokens/s on a cell all four finish is secondary.

Drive this from Arctic's DeepSpeed SFT launcher.
