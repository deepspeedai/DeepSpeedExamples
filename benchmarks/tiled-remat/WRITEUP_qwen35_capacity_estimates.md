# Qwen3.5 capacity estimates

## Executive summary

Gains are capacity divided by the gc baseline, at 140 GiB. Packed gains use the long-sweep crossings. Non-packed 9B and 27B, and the microbatch table, remain the stack formula. The sections below hold the absolute lengths and microbatch counts.

**Packed.** Equal 4k samples, one row, microbatch 1, tile budget 8192. The gain is the measured 140 GiB crossing divided by the gc crossing.

| Model | offload | tiling | tiling_offload |
|---|---:|---:|---:|
| 4B | 1.73× | 1.24× | 2.58× |
| 9B | 1.83× | 1.23× | 3.01× |
| 27B | ≥2.4× | ≥1.2× | 4.9× |

Offload is the larger single lever on 4B and 27B. 9B offload stays at the 8k-tile ratio: 384k returned no peak, so that arm was not retied. 27B gc, offload, and tiling are lower bounds from the longest fit still under 140 GiB, and 4.9× divides the 195k tiling_offload crossing by the 40k gc fit. The 8k-tile formula in section 1 is larger on tiling_offload because it holds GDN at one 8k sample. The measured 4k rows do not.

**Non-packed.** One sample. Sequence tiling has no boundary to cut, so tiling matches gc and tiling_offload matches offload.

| Model | offload | tiling | tiling_offload |
|---|---:|---:|---:|
| 4B | 1.73× | 1.00× | 1.73× |
| 9B | 1.82× | 1.00× | 1.82× |
| 27B | 2.63× | 1.00× | 2.63× |

4B offload uses the measured gc and offload crossings, 347k and 601k. 9B and 27B remain the formula in section 2.

**Microbatch scaling at 4k.** Tiling and tiling_offload are 2 shards. The last two columns are the ceiling, one live microbatch of mixer.

| Model | offload | tiling | tiling_offload | max tiling | max tiling_offload |
|---|---:|---:|---:|---:|---:|
| 4B | 1.82× | 1.34× | 3.39× | 2.03× | 24.7× |
| 9B | 1.83× | 1.35× | 3.43× | 2.02× | 27.6× |
| 27B | 2.88× | 1.25× | 5.25× | 1.50× | 39.1× |

More shards keep paying on tiling_offload and flatten quickly on tiling. Versus gc, tiling_offload at 4 and 8 shards is 5.95× / 9.61× on 4B, 6.09× / 10.0× on 9B, and 9.50× / 15.0× on 27B. Tiling at 8 shards is 1.81×, 1.80×, and 1.50×.

## Memory floor and baseline opts

ZeRO-2 GPU Adam, 8×H200. The limit is 140 GiB. The gains above are what offload and tiling add after the floor and the four opts below. Those four are on in every arm, including gc. Source sweep: `autorun/logs/job-20261003T220933Z.log`.

The floor is the per-GPU parameters plus GPU Adam state. It does not grow with sequence length or microbatch. It is the measured short-sequence peak minus the estimated activations at packed 8k, microbatch 1 (GDN + checkpoints + tiled FFN).

| Model | Measured peak (GiB) | 8k activations (GiB) | Floor (GiB) |
|---|---:|---:|---:|
| Qwen3.5-4B | 20.34 | 2.79 | 17.55 |
| Qwen3.5-9B | 41.78 plateau | 4.43 | 37.35 |
| Qwen3.5-27B | 112.82 plateau | 8.06 | 104.76 |

9B and 27B sit on that plateau through 16k and 8k–16k, so activations shorter than the plateau do not show up in the measured peak.

Baseline opts, already removed from the rising stack:

- **FlashAttention.** Varlen attention. The score matrix is not in the saved graph. Sweep C still records a linear full-attention workspace, 0.095 / 0.152 / 0.190 GiB per 1k tokens on 4B / 9B / 27B. GDN is larger (0.177 / 0.283 / 0.354), so the stack keeps GDN and does not add FlashAttention.
- **Liger.** Liger kernels are applied on the Qwen3.5 path used for this sweep.
- **Fused-CE.** Liger fused linear + cross-entropy (`--fused_ce`). The LM head does not materialize the token-by-vocab logit tensor, which would otherwise grow with every token.
- **TiledMLP, 4 shards.** Gate, up, and down are rematerialized one token shard at a time. The live FFN term is the unshared `2·B·S·(3I+H)` workspace divided by 4. GiB per 1k tokens at microbatch 1:

| Model | Unshared FFN | Tiled FFN |
|---|---:|---:|
| Qwen3.5-4B | 0.058 | 0.014 |
| Qwen3.5-9B | 0.078 | 0.020 |
| Qwen3.5-27B | 0.109 | 0.027 |

FlashAttention, Liger, fused-CE, and TiledMLP do not cut the checkpoint store or the live GDN rematerialize. Offload owns the store. Sequence and batch tiling own the live mixer. GDN itself runs through `fla`.

These are stack estimates. Measured peaks are cited only where a cell finished.

Arms:

- **gc** — gradient checkpointing
- **offload** — gc plus DeepSpeed activation offload
- **tiling** — tiled rematerialization, on the sequence axis or the microbatch axis as the section says
- **tiling_offload** — tiling plus activation offload

The stack is `floor + checkpoints + tiled FFN + live GDN`. Full attention is the smaller mixer (0.095 / 0.152 / 0.190 GiB per 1k tokens) and is not added. GDN is the Sweep C workspace on 4B, 0.177 GiB per 1k tokens, scaled by hidden size for 9B and 27B.

| Model | Floor (GiB) | GDN | Checkpoints | Tiled FFN | Stacked |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B | 17.55 | 0.177 | 0.156 | 0.014 | 0.347 |
| Qwen3.5-9B | 37.35 | 0.283 | 0.250 | 0.020 | 0.553 |
| Qwen3.5-27B | 104.76 | 0.354 | 0.625 | 0.027 | 1.006 |

Slopes are GiB per 1k tokens at microbatch 1. Shapes: 4B H=2560 I=9216 L=32, 9B H=4096 I=12288 L=32, 27B H=5120 I=17408 L=64. The floor and the baseline opts are in the section above.

Offload zeros the checkpoint slope. Sequence tiling, budget 8192, holds the live GDN at its 8k workspace (1.42 / 2.27 / 2.84 GiB) once the row is longer than 8k. Batch tiling leaves the checkpoints on the full microbatch and keeps one shard of the mixer.

## 1. Packed

One physical row, microbatch 1, equal 8k samples. gc and offload rematerialize the whole pack, so GDN grows with total tokens. tiling and tiling_offload keep GDN at one sample, which is the 8k tile. The count is the most 8k samples whose estimate stays at or under 140 GiB. The next sample crosses 140 GiB.

| Model | gc | offload | tiling | tiling_offload |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 44 × 8k = 352k | 80 × 8k = 640k | 88 × 8k = 704k | 1080 × 8k = 8640k |
| Qwen3.5-9B | 23 × 8k = 184k | 42 × 8k = 336k | 46 × 8k = 368k | 627 × 8k = 5016k |
| Qwen3.5-27B | 4 × 8k = 32k | 11 × 8k = 88k | 6 × 8k = 48k | 150 × 8k = 1200k |

The tiling columns assume the live mixer stays at one 8k sample, the Sweep C workspace. The measured 64k-to-128k slopes are from equal 4k samples with an 8192 budget, two samples per tile. Those tiling slopes are steeper: 0.295 GiB per 1k tokens on 4B and 0.403 on 9B. Those slopes reach 140 GiB at 421k and 272k. tiling_offload measured slopes, 0.158 and 0.185, reach 140 GiB at 783k and 595k.

The long sweep on the Qwen3.5 env uses the same 4k packing. Its 140 GiB crossings:

| Model | gc | offload | tiling | tiling_offload |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 347k | 601k | 431k | 895k |
| Qwen3.5-9B | 221k | — | 272k | 666k |
| Qwen3.5-27B | ≥40k | ≥96k | ≥48k | 195k |

4B gc is the 64k-to-128k slope. The new 256k peak is 107.5 GiB, and 384k runs out of memory. 4B offload is 70.0 GiB at 256k and 124.1 GiB at 512k (0.211 GiB per 1k tokens). 4B tiling is 129.1 GiB at 384k and out of memory at 512k (0.295 GiB per 1k tokens). 4B tiling_offload is 91.9 GiB at 512k and 126.0 GiB at 768k (0.133 GiB per 1k tokens). 9B gc repeats the 128k peak, 93.8 GiB, and runs out of memory at 256k. 9B offload has no new crossing: 95.5 GiB at 256k, no peak at 384k, out of memory at 512k. 9B tiling is 133.4 GiB at 256k and out of memory at 320k. 9B tiling_offload is 73.4 GiB at 256k and 116.7 GiB at 512k (0.169 GiB per 1k tokens). 27B gc is 127.2 GiB at 40k, offload is 127.9 GiB at 96k, and tiling is 129.1 GiB at 48k. The next cell on each of those arms returned no peak. 27B tiling_offload is 117.6 GiB at 96k and 125.2 GiB at 128k (0.236 GiB per 1k tokens).

27B checkpointing and tiling both fit at 32k (119.6 and 115.6 GiB) and run out of memory at 64k on the earlier sweep. Offload is 117.0 GiB at 64k; 128k did not record a peak. tiling_offload is still on the 112.9 GiB plateau at 64k, and 128k did not record a peak. The 1200k figure is the 8k-tile formula.

## 2. Non-packed

One contiguous sample, microbatch 1. Sequence tiling cannot cut inside that sample, so tiling matches gc and tiling_offload matches offload. GDN grows with the whole sample.

| Model | gc, tiling | offload, tiling_offload |
|---|---:|---:|
| Qwen3.5-4B | 353k | 641k |
| Qwen3.5-9B | 186k | 339k |
| Qwen3.5-27B | 35k | 92k |

Continuing the measured 64k-to-128k gc and offload slopes reaches 140 GiB at 347k and 587k on 4B, and at 221k and 431k on 9B. The long sweep keeps 4B gc at 347k and moves 4B offload to 601k. On 27B the formula crosses 140 GiB at 35k, while the earlier checkpointing peak is 119.6 GiB at 32k and out of memory at 64k. Offload at 64k is 117.0 GiB on that sweep. The long sweep still fits 27B gc at 40k and offload at 96k.

The linear crossing matches the packed stack. The packed table keeps only whole 8k samples, so 4B gc is 352k here and 353k above, and 27B offload is 88k in the packed table against a 92k crossing. A single sample has no tile boundary.

## 3. Microbatch scaling at 4k

Each microbatch is one 4k sample. Per-microbatch slopes are the per-1k-token slopes times 4.

| Model | GDN / mbs | Checkpoints / mbs | Tiled FFN / mbs |
|---|---:|---:|---:|
| Qwen3.5-4B | 0.708 | 0.624 | 0.056 |
| Qwen3.5-9B | 1.132 | 1.000 | 0.080 |
| Qwen3.5-27B | 1.416 | 2.500 | 0.108 |

Tiling is batch rematerialization. The live GDN is one shard of the microbatch. Checkpoints and tiled FFN still follow the full microbatch. A 4k row is one sequence tile under the 8k budget, so packing shorter samples into that row does not change these limits. gc and offload depend on total tokens. Where the 4k microbatch count is even, it is twice the packed 8k sample count. 27B offload is the odd case: 23 microbatches is 92k, and the packed grid stops at 11 × 8k.

At 2 shards, the largest microbatch at or under 140 GiB:

| Model | gc | offload | tiling | tiling_offload |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 88 | 160 | 118 | 298 |
| Qwen3.5-9B | 46 | 84 | 62 | 158 |
| Qwen3.5-27B | 8 | 23 | 10 | 42 |

The next microbatch crosses 140 GiB in every cell. gc and offload do not depend on shard count.

Shard count changes the tiling arms only. 4 shards leave a quarter of the mixer, 8 shards leave an eighth. The last column is the ceiling where the live mixer is a single microbatch, which is the smallest a shard can be.

| Model | Arm | 2 shards | 4 shards | 8 shards | one-microbatch mixer |
|---|---|---:|---:|---:|---:|
| 4B | tiling | 118 | 142 | 159 | 179 |
| 4B | tiling_offload | 298 | 524 | 846 | 2173 |
| 9B | tiling | 62 | 75 | 83 | 93 |
| 9B | tiling_offload | 158 | 280 | 462 | 1268 |
| 27B | tiling | 10 | 11 | 12 | 12 |
| 27B | tiling_offload | 42 | 76 | 120 | 313 |

On tiling, extra shards stop adding microbatches once the mixer is smaller than the checkpoint slope. 27B is at that ceiling by 8 shards. On tiling_offload the checkpoint slope is gone, so 4 and 8 shards keep adding microbatches. The leftover slope is the tiled FFN, which is why the one-microbatch column stays above 8 shards.
