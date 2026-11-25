# Vision encoder alone on one H100: Megatron-Bridge (Transformer Engine) against Hugging Face

The question: is Hugging Face's encoder cheaper per call than Megatron-Bridge's, which is built from
Transformer Engine modules, and why?

## Setup

- Model: the Qwen3.5-27B vision encoder (27 layers, width 1,152; the 9B's is the same but for the
  merger's output), random weights, bf16, train mode.
- Bridge's encoder is built exactly as its Qwen3VLModel does (`enc_bench.py build_bridge`,
  Megatron-Bridge v0.6.0, TE 2.18, NeMo 26.08 container).
- Hugging Face's uses SDPA attention, as the blog's rdsp runs did. It runs in rdsp's venv (torch
  2.13.0, transformers 5.17.0) and in the NeMo container (its torch, transformers 5.12.1).
- Images: 64 real CORD-v2 receipts through the Qwen3.5 processor capped at 1,000,000 pixels: 3,211
  patches per image on average, against 3,203 in the training rows (`grids.py`, `results/grids.log`,
  `results/round2/export_grids.txt`).
- Time: the median of 3 passes over the 64 images, synchronized per call, no profiler.
- GPU work, kernels, syncs and the per-part split come from a separate profiled pass of 16 images.
  The profiler slows the CPU side, so per-part CPU times are for comparison only.
- Hardware: round 1 on one H100 80GB (`results/hardware.txt`); round 2 on an 8× H100 node, four
  jobs at a time, one GPU each (`results/round2/hardware4.txt`).

## How to run

Run from `bench/encoder_test`, with the Qwen3.5-27B config files downloaded (no weights needed) and
`HF_HUB_OFFLINE=1`. First the image grids:

```bash
python grids.py results/grids.json 64
```

Bridge runs in the NeMo container:

```bash
C="docker run --gpus all --rm --ipc=host --shm-size=16g -v $HOME/hf:/root/.cache/huggingface \
  -v $HOME/rdsp:/rdsp -e HF_HOME=/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -w /rdsp/bench/encoder_test nvcr.io/nvidia/nemo:26.08"
$C python enc_bench.py --impl bridge --config Qwen/Qwen3.5-27B --images 1,2,4,8 \
  --grids results/grids.json --trace-dir results/traces --out results/bridge.json
```

Hugging Face runs in rdsp's venv:

```bash
python enc_bench.py --impl hf --hf-attn sdpa --label hf-sdpa-rdsp --config Qwen/Qwen3.5-27B \
  --images 1,2,4,8 --grids results/grids.json --trace-dir results/traces --out results/hf-sdpa-rdsp.json
```

The other runs add flags to these two commands:

| Run | Flags |
|---|---|
| Hugging Face in the container (`hf-sdpa-container`) | the Hugging Face command inside `$C`, `--modes fwd` |
| Every layer recomputed (`round2/*-rc`) | `--modes fwdbwd --recompute` |
| Timing repeats (`round2/*-r2`, `*-r3`) | `--profile-images 0` |
| Bridge's vision CUDA graphs (`round2/bridge-cg-*`) | `--images N --modes fwdbwd --cuda-graphs`, with `--recompute` for `bridge-cg-rc-*` |
| Hugging Face with per-layer CUDA graphs (`round2/hf-cg-*`, a prototype) | `--hf-attn varlen --hf-cuda-graphs --images N --modes fwdbwd` |
| The 35B-A3B encoder (`round2/*-35b`) | `--config Qwen/Qwen3.5-35B-A3B --images 1,8 --profile-images 0` |
| Bridge with the longest image length as an integer (`bridge-int-max-seqlen`) | `ENC_INT_MAX_SEQLEN=1`, `--images 1,8` |

Each run writes its JSON (`--out`); `python summarize.py results/*.json` makes the tables in
`results/summary.md` and `results/round2/summary4.md`. `sync_stacks.py --grids results/grids.json`
(in the container) lists the lines that make the CPU wait for the GPU (`results/sync_stacks.txt`).

## Every layer recomputed, as in the 27B training runs

Per image, one image per call, forward and backward. Time is the median of three runs; GPU work
is from the profiled run.

| | Bridge (TE) | Hugging Face, rdsp's venv |
|---|---|---|
| Time | 173 ms (172.3, 173.1, 175.6) | 114 ms (113.3, 114.8, 114.2) |
| GPU work | 73.8 ms | 62.4 ms |
| Time at 8 images per call | 75 ms (75.1, 74.6, 74.8) | 65 ms (64.6, 64.4, 64.9) |
| Time with Bridge's vision CUDA graphs | 78 ms (78.4, 78.5, 78.3) | |

Files: `round2/bridge-rc*.json`, `round2/hf-sdpa-rdsp-rc*.json`, `round2/bridge-cg-rc-n1-r*.json`.
Megatron-Bridge v0.6.0 has no setting that turns these graphs on for Qwen3.5 in training.

## Per image, one image per call, no recompute

| | Bridge (TE) | Hugging Face, rdsp's venv | Hugging Face, NeMo container |
|---|---|---|---|
| Forward | 60.5 ms | 25.8 ms | 25.9 ms |
| Forward + backward | 123.0 ms | 74.7 ms | (cuDNN error in backward) |
| Forward GPU work | 18.5 ms | 15.5 ms | 15.6 ms |
| Forward GPU kernels | 1,238 | 782 | 798 |
| Forward CPU waits for the GPU | 131 | 94 | 88 |

The same torch (container) gives Hugging Face the same time: the PyTorch version is not the cause.

Round 2's repeats of these runs (`round2/bridge-r*`, `round2/hf-sdpa-rdsp-r*`) ran on a different
machine: forward + backward 115-117 ms for Bridge and 69-70 ms for Hugging Face.

## Where the forward pass spends it, one image per call

| Part | Kernels: Bridge | Kernels: HF | GPU ms: Bridge | GPU ms: HF | CPU ms traced: Bridge | CPU ms traced: HF |
|---|---|---|---|---|---|---|
| Attention kernel call | 215 | 27 | 5.2 | 4.3 | 21.7 | 3.5 |
| Linear layers (TE's fuse the layer norm) | 162 | 108 + 54 norm | 4.4 | 3.8 + 0.6 | 24.8 | 8.4 + 3.6 |
| Rotary position embedding | 648 | 378 | 5.4 | 4.6 | 22.6 | 11.2 |
| Rest of attention, MLP, layer, outside | 213 | 215 | 3.5 | 2.2 | 23.0 | 20.4 |
| Total | 1,238 | 782 | 18.5 | 15.5 | 92.0 | 47.1 |

- Attention: TE's DotProductAttention runs 8 kernels and about 6x the CPU time per layer for the
  same fused attention kernel.
- Attention waits: TE's settings check (`utils.py:320 if sf != of`) and its fused call each read a
  value back from the GPU twice per layer (`results/sync_stacks.txt`), because Bridge passes the
  longest image length as a GPU tensor (`seqlens.max()`). Passing it as an integer removes 108 of
  131 waits but saves only 1.7 ms (60.5 to 58.8 ms): the waits are not the main cost.
- Linear layers: TE's linear modules cost about twice the CPU time of nn.Linear plus nn.LayerNorm.
- Rotary: Megatron Core's rotary code runs 24 kernels per layer (cos and sin recomputed for query and
  key in every layer, scale by 1.0, casts, a trailing concat); Hugging Face's 14.

## More images per call, no recompute

| Images per call | Bridge fwd | HF fwd | Bridge fwd+bwd | HF fwd+bwd | Ratio fwd+bwd |
|---|---|---|---|---|---|
| 1 | 60.5 | 25.8 | 123.0 | 74.7 | 1.65x |
| 2 | 35.4 | 20.7 | 76.8 | 57.7 | 1.33x |
| 4 | 23.9 | 17.7 | 61.0 | 52.6 | 1.16x |
| 8 | 21.9 | 16.6 | 57.1 | 50.5 | 1.13x |

ms per image. At 8 images per call both are close to their GPU work (Bridge 48.9 ms, HF 45.4 ms
forward + backward); what is left is Bridge's extra GPU work (more, smaller kernels).

## Other round-2 results

- 35B-A3B encoder, forward + backward, one image per call: Bridge 118.0 ms, Hugging Face 71.0 ms
  (`round2/*-35b.json`).
- Bridge's vision CUDA graphs without recompute: 61.4 ms at one image per call, but slower at more
  images per call (77, 114 and 188 ms per image at 2, 4 and 8), because each graph pads every image
  to one length (`round2/bridge-cg-n*.json`).
- rdsp's varlen vision attention: forward + backward 80.5 ms at one image per call, against SDPA's
  74.7 ms (`round2/hf-varlen-rdsp.json`; round 1's `results/hf-varlen-rdsp.log` failed to load it).
- Hugging Face with per-layer CUDA graphs, a prototype: 78 ms per image at one image per call
  without recompute, 98 ms with it (`round2/hf-cg-*.json`).
