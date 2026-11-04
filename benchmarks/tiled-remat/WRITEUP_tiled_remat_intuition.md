# Batch-Axis Tiled Rematerialization

*What the mechanism is and why it exists.*

## Motivation

After ZeRO, gradient checkpointing, and CPU offload, activation memory can still limit training. The relevant baseline is not a naive full-activation graph. It already uses FlashAttention, Liger fused linear+cross-entropy (no materialized logits), and TiledMLP. Those remove the usual long-sequence FFN and logit tensors.

Checkpointing then keeps layer inputs and recomputes the rest in backward; offload can move those **stored** inputs off the GPU. Backward still rematerializes a **full-microbatch live set** inside sequence-mixing layers. That workspace scales with microbatch and, at fixed microbatch, with sequence length. FlashAttention, Liger fused-CE, and TiledMLP do not cut it. If it is left unaddressed, the remaining knobs are more PCIe traffic or a smaller batch.

## Gap

Rematerialization is known to have a live set. What was missing is treating that term as the limiter on microbatch **after** those ALST-style reductions, under combined checkpointing and offload, and reducing it along the **batch** axis of sequence mixers (attention, Gated DeltaNet).

TiledMLP and Ulysses shard tokens or sequence. They are the right tools for long sequences at microbatch 1. They are the wrong axis when the bound is mixer rematerialization at microbatch greater than 1.

## Idea

Batch-axis tiled rematerialization: a full-microbatch `no_grad` forward, then rematerialize **one batch shard** in backward. Sequence mixers are independent across batch, so a shard is a valid rematerialize unit.

Saved inputs remain full-batch and can still be offloaded (`tiling_offload`). ZeRO gathers are held across the shard loop and gradients reduce once per backward, so this is not a loop of microbatch-1 steps. At microbatch 1 the method is a no-op.

## What this is meant to show

Measured on 8×H200, ZeRO-2 GPU Adam, FlashAttention, Liger fused cross-entropy, `fla`, and TiledMLP shards=4 (`autorun/logs/job-20261003T220933Z.log`). The peak decomposition later in this note is an older ZeRO-2 CPU-Adam ladder with TiledMLP off. Do not quote those GiB/microbatch slopes as this baseline.

1. Under this baseline the GPU-Adam floor dominates before sequence length does. Qwen3.5-27B stays at 112.8 GiB from packed 8k through 16k, in every mode. Qwen3.5-4B at 32k still grows with microbatch: checkpointing 27.3 / 38.8 / 61.7 GiB at microbatch 1 / 2 / 4. `ds_async` is 23.0 / 29.4 / 42.9 GiB.
2. Batch-axis tiling composes with TiledMLP. An inner last shard must not reduce while an outer batch tile is still open; `limit_grad_reduce` keeps `ds_grad_is_ready` false until both are on their last shard. The Oct 3 Qwen3.5 microbatch cells aborted before that gate and have no peak. The Oct 4 Qwen3 run, with TiledMLP shards=4, returned peaks: at microbatch 4, Qwen3-4B checkpointing 54.26 GiB and batch tiling 47.21 GiB, Qwen3-8B 57.34 and 52.81. Qwen3-32B still ran out of memory.
3. The arm that runs is packed whole-sample tiling, and it stacks with `ds_async`. See [WRITEUP_packed_seq_tiled_remat.md](WRITEUP_packed_seq_tiled_remat.md). `ds_async` removes more of the packed-length peak than tiling. Tiling does not raise tokens/s.

## Peak decomposition

Peak allocated memory is **floor + on-GPU checkpoint store + remat workspace**. Floor (parameters + optimizer) does not grow with microbatch. Store is the saved layer inputs; offload can move it to CPU. Workspace is the live set of a full-microbatch rematerialize; checkpointing and offload leave it intact.

On Qwen3.5-4B, ZeRO-2 + CPU Adam, 32k, 8×H200 (shards=2), each extra microbatch costs **20.8 GiB**. **16.1 GiB (77%) is remat workspace**; only **4.7 GiB (23%) is store**. That wrap is not one tensor class: **~14.3 GiB/mbs is mixer remainder** (GDN/FA + residuals); **~1.8 GiB/mbs is FFN** (`3·B·S·I + B·S·H` bf16, TiledMLP off). Offload can remove the 4.7. Tiling cuts the whole wrap **16.1 → 8.6 GiB/mbs** (mixer and FFN both along batch). Accounting: store = tiled peak − tiled_offload peak; workspace = peak − floor − on-GPU store; floor is the mbs intercept (~11.4 GiB). FFN is an architecture estimate; mixer is measured wrap minus that slice.

CUDA max allocated, GiB:

Microbatch 2 (all four modes measured):

- Checkpointing: 11.4 floor + 9.4 store + **28.7 mixer + 3.7 FFN** = 53.2
- Offload: 11.4 + 0 store + **28.4 mixer + 3.7 FFN** = 43.5
- Tiling: 11.4 + 9.4 store + **15.2 mixer + 1.9 FFN** = 37.8
- Tiling + offload: 11.4 + 0 store + **15.2 mixer + 1.9 FFN** = 28.5

Microbatch 4:

- Checkpointing: 11.4 + 18.8 store + **57.6 mixer + 7.4 FFN** = 95.1
- Offload: 11.4 + 0 + **57.0 mixer + 7.4 FFN** = 75.8
- Tiling: 11.4 + 18.8 + **30.3 mixer + 3.9 FFN** = 64.3
- Tiling + offload: 11.4 + 0 + **30.3 mixer + 3.9 FFN** = 45.6

The tall segment is mixer remat on every rung. FFN is the thin slice. Offload removes store; tiling halves the wrap; both compose. ZeRO-3 + GPU Adam on the same 4B/32k ladder has the same store (4.7 GiB/mbs) and the same ~78% workspace share; Z3 changes the floor, not which term scales with batch.

## Illustration

On Qwen3.5-4B with ZeRO-2 (Liger fused-CE, FlashAttention 2), that workspace slope is why tiling trains 128k where checkpointing and offload do not, and why tiling plus offload uniquely reaches 192k.

Maximum microbatch at 32k:

- Checkpointing: 4
- Activation offload: 6
- Tiling: 8
- Tiling + offload: 12

On Qwen3.5-35B-A3B (AutoEP, ZeRO-2, GPU Adam), the combined mode uniquely fits 64k and microbatch 16 in current sweeps.

Whether the same tile can be the expert-parallel communication grain is a separate question.

## Next direction: packed sequences at `mbs=1`

Batch-axis tiling does not help physical `mbs=1`, and `mbs=1` plus GAS already matches or beats tiled `mbs>1` on throughput. The leftover capacity problem is mixer remat on a **packed** long sequence: many independent shorter samples in one row, boundaries in `cu_seqlens`. That axis is legal for the same reason dim 0 is legal — mixers do not cross sample boundaries — and it is what TiledMLP cannot cut. See [WRITEUP_packed_seq_tiled_remat.md](WRITEUP_packed_seq_tiled_remat.md).
