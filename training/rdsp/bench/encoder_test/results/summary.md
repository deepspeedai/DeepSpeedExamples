## Environments

- bridge: {"torch": "2.13.0a0+8145d630e8.nv26.06", "transformers": "5.12.1", "python": "3.12.3", "gpu": "NVIDIA H100 80GB HBM3", "params": 460730096, "transformer_engine": "2.18.0+15288857"}
- hf-sdpa-container: {"torch": "2.13.0a0+8145d630e8.nv26.06", "transformers": "5.12.1", "python": "3.12.3", "gpu": "NVIDIA H100 80GB HBM3", "params": 460730096, "transformer_engine": "2.18.0+15288857"}
- hf-sdpa-rdsp: {"torch": "2.13.0+cu130", "transformers": "5.17.0", "python": "3.12.16", "gpu": "NVIDIA H100 80GB HBM3", "params": 460730096}

## fwd: per image

| Encoder | Images per call | Time (ms) | GPU busy (ms) | GPU share | Kernels | Syncs | Traced time (ms) |
|---|---|---|---|---|---|---|---|
| bridge | 1 | 60.5 | 18.5 | 31% | 1,238 | 131 | 92.8 |
| bridge | 2 | 35.4 | 17.5 | 49% | 631 | 72 | 49.1 |
| bridge | 4 | 23.9 | 16.8 | 70% | 325 | 43 | 26.7 |
| bridge | 8 | 21.9 | 16.3 | 75% | 171 | 29 | 21.4 |
| hf-sdpa-container | 1 | 25.9 | 15.6 | 60% | 798 | 88 | 49.9 |
| hf-sdpa-container | 2 | 20.9 | 15.0 | 72% | 454 | 44 | 28.0 |
| hf-sdpa-container | 4 | 18.2 | 14.6 | 80% | 261 | 22 | 19.4 |
| hf-sdpa-container | 8 | 17.1 | 14.4 | 84% | 165 | 11 | 17.3 |
| hf-sdpa-rdsp | 1 | 25.8 | 15.5 | 60% | 782 | 94 | 47.8 |
| hf-sdpa-rdsp | 2 | 20.7 | 14.9 | 72% | 424 | 47 | 27.3 |
| hf-sdpa-rdsp | 4 | 17.7 | 14.5 | 82% | 228 | 24 | 18.5 |
| hf-sdpa-rdsp | 8 | 16.6 | 14.3 | 86% | 130 | 12 | 16.5 |

## fwdbwd: per image

| Encoder | Images per call | Time (ms) | GPU busy (ms) | GPU share | Kernels | Syncs | Traced time (ms) |
|---|---|---|---|---|---|---|---|
| bridge | 1 | 123.0 | 56.4 | 46% | 2,880 | 185 | 171.9 |
| bridge | 2 | 76.8 | 52.6 | 69% | 1,455 | 100 | 92.8 |
| bridge | 4 | 61.0 | 50.1 | 82% | 737 | 57 | 62.0 |
| bridge | 8 | 57.1 | 48.9 | 86% | 378 | 35 | 54.7 |
| hf-sdpa-rdsp | 1 | 74.7 | 49.5 | 66% | 2,467 | 94 | 111.7 |
| hf-sdpa-rdsp | 2 | 57.7 | 47.7 | 83% | 1,308 | 47 | 64.8 |
| hf-sdpa-rdsp | 4 | 52.6 | 46.3 | 88% | 711 | 24 | 51.4 |
| hf-sdpa-rdsp | 8 | 50.5 | 45.4 | 90% | 412 | 12 | 48.2 |

## fwd, 1 image per call: kernels per image, by part

| Part | bridge | hf-sdpa-container | hf-sdpa-rdsp |
|---|---|---|---|
| rotary | 648 | 432 | 378 |
| core_attn | 215 | 27 | 27 |
| linear | 162 | 108 | 108 |
| norm |  | 54 | 54 |
| attn_other |  | 27 | 27 |
| mlp_other | 54 | 27 | 27 |
| layer_other | 108 | 54 | 54 |
| outside | 51 | 69 | 107 |

## fwd, 1 image per call: GPU ms per image, by part

| Part | bridge | hf-sdpa-container | hf-sdpa-rdsp |
|---|---|---|---|
| rotary | 5.40 | 4.82 | 4.58 |
| core_attn | 5.19 | 4.33 | 4.32 |
| linear | 4.41 | 3.80 | 3.82 |
| norm |  | 0.58 | 0.58 |
| attn_other |  | 0.04 | 0.04 |
| mlp_other | 1.46 | 0.52 | 0.52 |
| layer_other | 0.86 | 0.34 | 0.34 |
| outside | 1.14 | 1.21 | 1.27 |

## fwd, 1 image per call: CPU ms (traced, forward thread, exclusive) per image, by part

| Part | bridge | hf-sdpa-container | hf-sdpa-rdsp |
|---|---|---|---|
| rotary | 22.62 | 12.57 | 11.22 |
| core_attn | 21.70 | 3.52 | 3.47 |
| linear | 24.79 | 8.93 | 8.36 |
| norm |  | 3.70 | 3.59 |
| attn_other | 4.68 | 11.17 | 10.59 |
| mlp_other | 3.35 | 2.02 | 1.90 |
| layer_other | 7.38 | 3.86 | 3.62 |
| outside | 7.52 | 3.54 | 4.35 |
