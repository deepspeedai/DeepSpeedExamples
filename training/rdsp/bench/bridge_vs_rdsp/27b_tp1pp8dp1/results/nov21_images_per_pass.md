# 27B, eight stages of one H100: images per microbatch, and each side's best

Qwen3.5-27B, TP1 x PP8 x DP1 on one 8× H100 node, 64 CORD-v2 rows a step (Megatron-Bridge's
export, replayed by rdsp), every row padded to 2,048, BF16 gradient accumulation. Settings
otherwise as the blog's runs (`scripts/run.sh`, `scripts/rdsp.sh`, `scripts/rdsp_tune.sh`).
Median of steps 6-50. Bridge runs at the cut rdsp measured for the same images per microbatch.
Logs in `../logs/`.

The runs come from two machines of the same type: `*-nov21fix-*` logs from one, `*-nov21-*` logs
from the other. Compare numbers within one table.

## Each side's best setting per images per microbatch, one machine

| Images per microbatch | DeepSpeed (rdsp) | Megatron-Bridge | Faster |
|---|---|---|---|
| 1 | 10.66 s | 15.13 s | DeepSpeed 1.42x |
| 2 | **9.28 s** | **9.78 s** | DeepSpeed 1.05x |
| 4 | 11.33 s | 10.47 s | Bridge 1.08x |
| 8 | out of memory (stage 0, 15 GiB allocation) | out of memory (last stage) | neither fits |

Logs: `rdsp-nov21fix-rows{1,2,4,8}-best.log`, `bridge-nov21fix-mbs{1,2,4,8}-best.log`.

Each side's best overall is 2 per microbatch: DeepSpeed 9.28 s against Bridge 9.78 s, 1.05x.

rdsp's setting is recompute per stage (`RC_PER`: blocks recomputed, counted at the calibration
cut): `6,8,7,7,6,5,3,0` at 1 and 2 per microbatch and `6,9,8,8,8,7,5,4` at 4. The calibration cut
gives stages 0-7 6, 9, 9, 9, 9, 9, 8 and 5 blocks, so at 1 and 2 per microbatch they keep 0, 1, 2,
2, 3, 4, 5 and 5 blocks' activations. At 8 per microbatch every block is recomputed. Bridge
recomputes every layer at every size; its lighter options did not fit (below).

## Every layer recomputed on both, the other machine

| Images per microbatch | DeepSpeed | Bridge | |
|---|---|---|---|
| 1 | 11.09 s | 14.79 s | DeepSpeed 1.33x |
| 2 | 10.03 s | 9.90 s | Bridge 1.01x |
| 4 | 11.24 s | 10.27 s | Bridge 1.09x |

Logs: `rdsp-nov21-rows{1,2,4}.log`, `bridge-nov21-mbs{1,2,4}.log`.

## What was tried (16-step trials at 2 per microbatch, steps 6-16)

| Trial | Result | Log |
|---|---|---|
| rdsp, decoder blocks compiled (Liger off) | 11.36 s: 13% slower per stage (calibration 7.6 against 6.7 s) | `rdsp-nov21fix-rows2-compile.log` |
| rdsp, recompute 6 of 9 blocks on every stage | out of memory on stages 1-2 (7 and 6 microbatches in flight) | `rdsp-nov21fix-rows2-rc6.log` |
| rdsp, recompute per stage (`RC_PER=6,8,7,7,6,5,3,0`) | 9.45 s (50 steps: 9.28 s) | `rdsp-nov21fix-rows2-per1.log` |
| Bridge, `recompute_method=block` with 9 and 8 layers | out of memory on stage 0: the vision encoder takes the language config's recompute setting (`get_vision_model_config`) | `bridge-nov21fix-mbs2-block{9,8}.log` |
| Bridge, selective recompute | out of memory (stages 0-2) | `bridge-nov21fix-mbs2-selective.log` |

Bridge's vision CUDA graphs cannot be turned on for Qwen3.5 in v0.6.0: the Qwen3.5 provider has
no `vision_cuda_graph_impl`.

## Bridge at 1 per microbatch on four machines

The same Bridge setting (every layer recomputed, rdsp's measured cut of that run) on four machines
of the same type: 14.34 s (`bridge-gap-cut1.log`), 14.45 s (`bridge-nov11.log`), 14.79 s
(`bridge-nov21-mbs1.log`) and 15.13 s (`bridge-nov21fix-mbs1-best.log`).

## Why

- 1 per microbatch: stage 0 (vision encoder + 1 layer) paces both. Bridge's encoder costs more per
  image on the CPU: 173 against 114 ms per image with recompute (`../../../encoder_test/results.md`).
- 2 per microbatch: the language stages pace. rdsp's per-stage recompute lets late stages (fewer
  microbatches in flight) keep activations. Bridge cannot recompute its language layers less
  without its vision encoder also recomputing less, and that does not fit stage 0.
- 4 per microbatch: rdsp's `--profile` run with every block recomputed
  (`../logs/rdsp-nov21fix-rows4-profile-all.log`, 16 steps) shows the busiest stage, stage 6 (10
  layers), at 7.38 s of an 11.4 s step. The 1F1B schedule needs 10.6 s at that stage, plus 0.25 s
  for the optimizer step, so about 0.6 s is pipeline overhead; 3 of the 16 steps took 17-18 s.
  Bridge at 10.47 s sits at about its floor. rdsp's measured cut leaves stage 7 at 5.42 s against
  7.38 s.
- 8 per microbatch: 8 microbatches for 8 stages, so the pipeline is idle 47% of each step; neither
  fits in memory.
