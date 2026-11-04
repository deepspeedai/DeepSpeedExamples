# Qwen3 capacity estimates

## Executive summary

Gains are capacity divided by the gc baseline, at 140 GiB. Packed gains use the long-sweep crossings. Non-packed 32B, and the microbatch table, remain the stack formula. The sections below hold the absolute lengths and microbatch counts.

**Packed.** Equal 4k samples, one row, microbatch 1, tile budget 8192. The gain is the measured 140 GiB crossing divided by the gc crossing.

| Model | offload | tiling | tiling_offload |
|---|---:|---:|---:|
| 4B | 2.24× | 1.19× | 3.49× |
| 8B | 2.50× | 1.12× | 3.37× |

Offload is the larger single lever, because checkpoints are the steep slope. On the measured 4k rows, tiling_offload is 3.49× on 4B and 3.37× on 8B. The 8k-tile formula in section 1 is larger because it holds full attention at one 8k sample. On 32B, gc and offload both ran out of memory at packed 4k and did not record a peak, so this sweep has no 32B gain. The 32B floor remains the 8B floor scaled by 31.98B/7.57B.

**Non-packed.** One sample. Sequence tiling has no boundary to cut, so tiling matches gc and tiling_offload matches offload. FlashAttention is on, so these multiples use the linear full-attention workspace.

| Model | offload | tiling | tiling_offload |
|---|---:|---:|---:|
| 4B | 2.24× | 1.00× | 2.24× |
| 8B | 2.50× | 1.00× | 2.50× |
| 32B | 4.71× | 1.00× | 4.71× |

4B and 8B offload use the measured gc and offload crossings: 416k and 932k on 4B, 242k and 605k on 8B. 32B remains the scaled-floor formula in section 2. Packed 4k gc and offload ran out of memory on that model.

**Microbatch scaling at 4k.** Tiling and tiling_offload are 2 shards. The last two columns are the ceiling, one live microbatch of mixer.

| Model | offload | tiling | tiling_offload | max tiling | max tiling_offload |
|---|---:|---:|---:|---:|---:|
| 4B | 3.20× | 1.14× | 5.39× | 1.33× | 17.0× |
| 8B | 3.32× | 1.15× | 5.73× | 1.35× | 20.4× |
| 32B | 8.00× | 1.00× | 12.0× | 1.00× | 32.0× |

More shards keep paying on tiling_offload and add little on tiling. Versus gc, tiling_offload at 4 and 8 shards is 8.19× / 11.0× on 4B, 8.94× / 12.4× on 8B, and 18.0× / 24.0× on 32B. Tiling at 8 shards is 1.28×, 1.30×, and 1.00×.

## Memory floor and baseline opts

ZeRO-2 GPU Adam, 8×H200. The limit is 140 GiB. The gains above are what offload and tiling add after the floor and the four opts below. Those four are on in every arm, including gc. Source sweep: `autorun/logs/job-20261004T175315Z.log`.

The floor is the per-GPU parameters plus GPU Adam state. It does not grow with sequence length or microbatch. On 4B and 8B it is the measured short-sequence peak minus the estimated activations at packed 8k, microbatch 1 (full attention + checkpoints + tiled FFN).

| Model | Measured peak (GiB) | 8k activations (GiB) | Floor (GiB) |
|---|---:|---:|---:|
| Qwen3-4B | 18.10 | 2.04 | 16.06 |
| Qwen3-8B | 35.05 plateau | 3.23 | 31.82 |
| Qwen3-32B | no peak | — | 134.46 |

The 32B floor is the 8B floor times 31.98B/7.57B. Every measured 32B cell ran out of memory, including packed 8k with offload, so 134.46 GiB is a parameter-ratio scale rather than a measured plateau. 8B sits on the 35.05 plateau through 16k.

Baseline opts, already removed from the rising stack:

- **FlashAttention.** Varlen attention. The score matrix is not in the saved graph. Sweep C on 4B records the linear rematerialize workspace that the stack does use: 0.065 GiB per 1k tokens, 67744 bytes per token, measured at 8k, 32k, and 64k. 8B and 32B scale that rate by hidden size, 4096/2560 and 5120/2560, to 0.103 and 0.129. There is no GDN.
- **Liger.** Liger kernels are applied on the Qwen3 path used for this sweep (`apply_liger_kernel_to_qwen3`).
- **Fused-CE.** Liger fused linear + cross-entropy (`--fused_ce`). The LM head does not materialize the token-by-vocab logit tensor. Qwen3 vocab is 151936, so that tensor would be about 0.29 GiB per 1k tokens at microbatch 1.
- **TiledMLP, 4 shards.** Gate, up, and down are rematerialized one token shard at a time. The live FFN term is the unshared `2·B·S·(3I+H)` workspace divided by 4. GiB per 1k tokens at microbatch 1:

| Model | Unshared FFN | Tiled FFN |
|---|---:|---:|
| Qwen3-4B | 0.061 | 0.015 |
| Qwen3-8B | 0.078 | 0.020 |
| Qwen3-32B | 0.156 | 0.039 |

FlashAttention, Liger, fused-CE, and TiledMLP do not cut the checkpoint store. TiledMLP does not cut the full-attention workspace. Offload owns the store. Sequence and batch tiling own the live attention rematerialize.

These are stack estimates. Measured peaks are cited only where a cell finished.

Arms:

- **gc** — gradient checkpointing
- **offload** — gc plus DeepSpeed activation offload
- **tiling** — tiled rematerialization, on the sequence axis or the microbatch axis as the section says
- **tiling_offload** — tiling plus activation offload

Qwen3 has no GDN. The stack is `floor + checkpoints + tiled FFN + live full attention`. Full attention is the Sweep C workspace on 4B, 0.065 GiB per 1k tokens, scaled by hidden size for 8B and 32B. That term is the linear rematerialize workspace. FlashAttention keeps the score matrix out of the saved graph, so a single long sample follows this linear rate.

| Model | Floor (GiB) | Full attention | Checkpoints | Tiled FFN | Stacked |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B | 16.06 | 0.065 | 0.176 | 0.015 | 0.256 |
| Qwen3-8B | 31.82 | 0.103 | 0.281 | 0.020 | 0.404 |
| Qwen3-32B | 134.46 | 0.129 | 0.625 | 0.039 | 0.793 |

Slopes are GiB per 1k tokens at microbatch 1. Shapes: 4B H=2560 I=9728 L=36, 8B H=4096 I=12288 L=36, 32B H=5120 I=25600 L=64. The floor and the baseline opts are in the section above.

Offload zeros the checkpoint slope. Sequence tiling, budget 8192, holds full attention at its 8k workspace (0.52 / 0.83 / 1.03 GiB) once the row is longer than 8k. Batch tiling leaves the checkpoints on the full microbatch and keeps one shard of the mixer.

## 1. Packed

One physical row, microbatch 1, equal 8k samples. gc and offload rematerialize the whole pack, so full attention grows with total tokens. tiling and tiling_offload keep full attention at one sample, which is the 8k tile. The count is the most 8k samples whose estimate stays at or under 140 GiB. The next sample crosses 140 GiB.

| Model | gc | offload | tiling | tiling_offload |
|---|---:|---:|---:|---:|
| Qwen3-4B | 60 × 8k = 480k | 193 × 8k = 1544k | 80 × 8k = 640k | 1028 × 8k = 8224k |
| Qwen3-8B | 33 × 8k = 264k | 109 × 8k = 872k | 44 × 8k = 352k | 670 × 8k = 5360k |
| Qwen3-32B | no 8k fit | 4 × 8k = 32k | no 8k fit | 14 × 8k = 112k |

On 32B one 8k sample is 140.8 GiB, so gc and tiling have no row under the line. Every measured 32B cell, including 8k with offload, ran out of memory. The 32k and 112k figures sit on the scaled floor.

The tiling columns assume the live mixer stays at one 8k sample, the Sweep C workspace. The measured 64k-to-128k slopes are from equal 4k samples with an 8192 budget, two samples per tile. Those tiling slopes are steeper: 0.260 GiB per 1k tokens on 4B and 0.415 on 8B. Those slopes reach 140 GiB at 482k and 271k. tiling_offload measured slopes, 0.089 and 0.142, reach 140 GiB at 1412k and 780k.

The long sweep on the Qwen3.5 env uses the same 4k packing. Its 140 GiB crossings:

| Model | gc | offload | tiling | tiling_offload |
|---|---:|---:|---:|---:|
| Qwen3-4B | 416k | 932k | 494k | 1450k |
| Qwen3-8B | 242k | 605k | 271k | 815k |
| Qwen3-32B | none | none | — | — |

4B gc is 93.8 GiB at 256k and 133.3 GiB at 384k (0.308 GiB per 1k tokens), and 512k runs out of memory. 4B offload is 50.0 GiB at 256k and 85.2 GiB at 512k (0.138 GiB per 1k tokens). 4B tiling is 81.2 GiB at 256k and 114.4 GiB at 384k (0.259 GiB per 1k tokens). 4B tiling_offload is 37.5 GiB at 256k and 60.1 GiB at 512k (0.088 GiB per 1k tokens); 1024k returned no peak. 8B gc repeats the 128k peak, 87.0 GiB, and runs out of memory at 256k. 8B offload is 76.4 GiB at 256k and 125.1 GiB at 512k (0.190 GiB per 1k tokens). 8B tiling is 133.8 GiB at 256k and out of memory at 320k. 8B tiling_offload is 63.8 GiB at 256k and 100.0 GiB at 512k (0.141 GiB per 1k tokens). 32B gc and offload run out of memory at 4k with no peak. Tiling was not run.

## 2. Non-packed

One contiguous sample, microbatch 1. Sequence tiling cannot cut inside that sample, so tiling matches gc and tiling_offload matches offload. Full attention grows with the whole sample, using the linear Sweep C rate.

| Model | gc, tiling | offload, tiling_offload |
|---|---:|---:|
| Qwen3-4B | 484k | 1549k |
| Qwen3-8B | 268k | 880k |
| Qwen3-32B | 7k | 33k |

Continuing the measured 64k-to-128k gc and offload slopes reaches 140 GiB at 406k and 910k on 4B, and at 242k and 605k on 8B. The long sweep moves 4B gc to 416k and 4B offload to 932k, and keeps 8B gc at 242k and 8B offload at 605k. 32B has no measured peak. The formula crosses 140 GiB at 7k, so the formula places 4k under the line (137.6 GiB) and 8k over it (140.8 GiB). Packed 4k gc and offload both ran out of memory on the long sweep.

The linear crossing matches the packed stack. The packed table keeps only whole 8k samples, so 4B gc is 480k here and 484k above, and 32B gc has no packed row under 140 GiB. A single sample has no tile boundary. With FlashAttention on, this linear table is the limit for that sample.

## 3. Microbatch scaling at 4k

Each microbatch is one 4k sample. Per-microbatch slopes are the per-1k-token slopes times 4.

| Model | Full attention / mbs | Checkpoints / mbs | Tiled FFN / mbs |
|---|---:|---:|---:|
| Qwen3-4B | 0.260 | 0.704 | 0.060 |
| Qwen3-8B | 0.412 | 1.124 | 0.080 |
| Qwen3-32B | 0.516 | 2.500 | 0.156 |

Tiling is batch rematerialization. The live attention workspace is one shard of the microbatch. Checkpoints and tiled FFN still follow the full microbatch. A 4k row is one sequence tile under the 8k budget, so packing shorter samples into that row does not change these limits. gc and offload depend on total tokens. Where the 4k microbatch count is even, it is twice the packed 8k sample count. 4B offload is the odd case: 387 microbatches is 1548k, and the packed grid stops at 193 × 8k.

At 2 shards, the largest microbatch at or under 140 GiB:

| Model | gc | offload | tiling | tiling_offload |
|---|---:|---:|---:|---:|
| Qwen3-4B | 121 | 387 | 138 | 652 |
| Qwen3-8B | 66 | 219 | 76 | 378 |
| Qwen3-32B | 1 | 8 | 1 | 12 |

The next microbatch crosses 140 GiB in every cell. gc and offload do not depend on shard count. The 32B counts use the scaled floor.

Shard count changes the tiling arms only. 4 shards leave a quarter of the mixer, 8 shards leave an eighth. The last column is the ceiling where the live mixer is a single microbatch, which is the smallest a shard can be.

| Model | Arm | 2 shards | 4 shards | 8 shards | one-microbatch mixer |
|---|---|---:|---:|---:|---:|
| 4B | tiling | 138 | 149 | 155 | 161 |
| 4B | tiling_offload | 652 | 991 | 1337 | 2061 |
| 8B | tiling | 76 | 82 | 86 | 89 |
| 8B | tiling_offload | 378 | 590 | 821 | 1347 |
| 32B | tiling | 1 | 1 | 1 | 1 |
| 32B | tiling_offload | 12 | 18 | 24 | 32 |

On tiling, extra shards stop adding microbatches once the mixer is smaller than the checkpoint slope. 32B stays at 1 for every shard count: two microbatches of checkpoints put the scaled floor over 140 GiB. On tiling_offload the checkpoint slope is gone, so 4 and 8 shards keep adding microbatches. The leftover slope is the tiled FFN, which is why the one-microbatch column stays above 8 shards.
