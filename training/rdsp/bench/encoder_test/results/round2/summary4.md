## Environments

- bridge-35b: {"torch": "2.13.0a0+8145d630e8.nv26.06", "transformers": "5.12.1", "python": "3.12.3", "gpu": "NVIDIA H100 80GB HBM3", "params": 446571248, "transformer_engine": "2.18.0+15288857"}
- bridge-cg-n1-r2: {"torch": "2.13.0a0+8145d630e8.nv26.06", "transformers": "5.12.1", "python": "3.12.3", "gpu": "NVIDIA H100 80GB HBM3", "params": 460730096, "transformer_engine": "2.18.0+15288857"}
- bridge-cg-n1-r3: {"torch": "2.13.0a0+8145d630e8.nv26.06", "transformers": "5.12.1", "python": "3.12.3", "gpu": "NVIDIA H100 80GB HBM3", "params": 460730096, "transformer_engine": "2.18.0+15288857"}
- bridge-cg-n1: {"torch": "2.13.0a0+8145d630e8.nv26.06", "transformers": "5.12.1", "python": "3.12.3", "gpu": "NVIDIA H100 80GB HBM3", "params": 460730096, "transformer_engine": "2.18.0+15288857"}
- bridge-cg-n2: {"torch": "2.13.0a0+8145d630e8.nv26.06", "transformers": "5.12.1", "python": "3.12.3", "gpu": "NVIDIA H100 80GB HBM3", "params": 460730096, "transformer_engine": "2.18.0+15288857"}
- bridge-cg-n4: {"torch": "2.13.0a0+8145d630e8.nv26.06", "transformers": "5.12.1", "python": "3.12.3", "gpu": "NVIDIA H100 80GB HBM3", "params": 460730096, "transformer_engine": "2.18.0+15288857"}
- bridge-cg-n8: {"torch": "2.13.0a0+8145d630e8.nv26.06", "transformers": "5.12.1", "python": "3.12.3", "gpu": "NVIDIA H100 80GB HBM3", "params": 460730096, "transformer_engine": "2.18.0+15288857"}
- bridge-r2: {"torch": "2.13.0a0+8145d630e8.nv26.06", "transformers": "5.12.1", "python": "3.12.3", "gpu": "NVIDIA H100 80GB HBM3", "params": 460730096, "transformer_engine": "2.18.0+15288857"}
- bridge-r3: {"torch": "2.13.0a0+8145d630e8.nv26.06", "transformers": "5.12.1", "python": "3.12.3", "gpu": "NVIDIA H100 80GB HBM3", "params": 460730096, "transformer_engine": "2.18.0+15288857"}
- bridge-rc-r2: {"torch": "2.13.0a0+8145d630e8.nv26.06", "transformers": "5.12.1", "python": "3.12.3", "gpu": "NVIDIA H100 80GB HBM3", "params": 460730096, "transformer_engine": "2.18.0+15288857"}
- bridge-rc-r3: {"torch": "2.13.0a0+8145d630e8.nv26.06", "transformers": "5.12.1", "python": "3.12.3", "gpu": "NVIDIA H100 80GB HBM3", "params": 460730096, "transformer_engine": "2.18.0+15288857"}
- bridge-rc: {"torch": "2.13.0a0+8145d630e8.nv26.06", "transformers": "5.12.1", "python": "3.12.3", "gpu": "NVIDIA H100 80GB HBM3", "params": 460730096, "transformer_engine": "2.18.0+15288857"}
- hf-sdpa-rdsp-35b: {"torch": "2.13.0+cu130", "transformers": "5.17.0", "python": "3.12.16", "gpu": "NVIDIA H100 80GB HBM3", "params": 446571248}
- hf-sdpa-rdsp-r2: {"torch": "2.13.0+cu130", "transformers": "5.17.0", "python": "3.12.16", "gpu": "NVIDIA H100 80GB HBM3", "params": 460730096}
- hf-sdpa-rdsp-r3: {"torch": "2.13.0+cu130", "transformers": "5.17.0", "python": "3.12.16", "gpu": "NVIDIA H100 80GB HBM3", "params": 460730096}
- hf-sdpa-rdsp-rc-r2: {"torch": "2.13.0+cu130", "transformers": "5.17.0", "python": "3.12.16", "gpu": "NVIDIA H100 80GB HBM3", "params": 460730096}
- hf-sdpa-rdsp-rc-r3: {"torch": "2.13.0+cu130", "transformers": "5.17.0", "python": "3.12.16", "gpu": "NVIDIA H100 80GB HBM3", "params": 460730096}
- hf-sdpa-rdsp-rc: {"torch": "2.13.0+cu130", "transformers": "5.17.0", "python": "3.12.16", "gpu": "NVIDIA H100 80GB HBM3", "params": 460730096}
- hf-varlen-rdsp: {"torch": "2.13.0+cu130", "transformers": "5.17.0", "python": "3.12.16", "gpu": "NVIDIA H100 80GB HBM3", "params": 460730096}

## fwd: per image

| Encoder | Images per call | Time (ms) | GPU busy (ms) | GPU share | Kernels | Syncs | Traced time (ms) |
|---|---|---|---|---|---|---|---|
| bridge-35b | 1 | 64.3 |  |  |  |  |  |
| bridge-35b | 8 | 21.8 |  |  |  |  |  |
| bridge-r2 | 1 | 63.9 |  |  |  |  |  |
| bridge-r2 | 2 | 33.2 |  |  |  |  |  |
| bridge-r2 | 4 | 23.7 |  |  |  |  |  |
| bridge-r2 | 8 | 21.9 |  |  |  |  |  |
| bridge-r3 | 1 | 63.2 |  |  |  |  |  |
| bridge-r3 | 2 | 33.6 |  |  |  |  |  |
| bridge-r3 | 4 | 23.8 |  |  |  |  |  |
| bridge-r3 | 8 | 21.8 |  |  |  |  |  |
| hf-sdpa-rdsp-35b | 1 | 26.2 |  |  |  |  |  |
| hf-sdpa-rdsp-35b | 8 | 16.5 |  |  |  |  |  |
| hf-sdpa-rdsp-r2 | 1 | 26.8 |  |  |  |  |  |
| hf-sdpa-rdsp-r2 | 2 | 20.0 |  |  |  |  |  |
| hf-sdpa-rdsp-r2 | 4 | 17.5 |  |  |  |  |  |
| hf-sdpa-rdsp-r2 | 8 | 16.5 |  |  |  |  |  |
| hf-sdpa-rdsp-r3 | 1 | 26.3 |  |  |  |  |  |
| hf-sdpa-rdsp-r3 | 2 | 20.1 |  |  |  |  |  |
| hf-sdpa-rdsp-r3 | 4 | 17.5 |  |  |  |  |  |
| hf-sdpa-rdsp-r3 | 8 | 16.6 |  |  |  |  |  |
| hf-varlen-rdsp | 1 | 26.4 | 15.4 | 58% | 782 | 94 | 50.5 |
| hf-varlen-rdsp | 2 | 21.1 | 14.8 | 70% | 424 | 47 | 27.7 |
| hf-varlen-rdsp | 4 | 17.9 | 14.5 | 81% | 228 | 24 | 18.9 |
| hf-varlen-rdsp | 8 | 16.7 | 14.4 | 86% | 130 | 12 | 16.6 |

## fwdbwd: per image

| Encoder | Images per call | Time (ms) | GPU busy (ms) | GPU share | Kernels | Syncs | Traced time (ms) |
|---|---|---|---|---|---|---|---|
| bridge-35b | 1 | 118.0 |  |  |  |  |  |
| bridge-35b | 8 | 57.0 |  |  |  |  |  |
| bridge-cg-n1-r2 | 1 | 61.6 |  |  |  |  |  |
| bridge-cg-n1-r3 | 1 | 61.0 |  |  |  |  |  |
| bridge-cg-n1 | 1 | 61.4 | 54.7 | 89% | 1,501 | 20 | 64.1 |
| bridge-cg-n2 | 2 | 77.1 | 71.7 | 93% | 764 | 17 | 78.3 |
| bridge-cg-n4 | 4 | 114.0 | 108.4 | 95% | 391 | 16 | 113.9 |
| bridge-cg-n8 | 8 | 188.1 | 183.6 | 98% | 205 | 15 | 188.2 |
| bridge-r2 | 1 | 116.8 |  |  |  |  |  |
| bridge-r2 | 2 | 75.8 |  |  |  |  |  |
| bridge-r2 | 4 | 60.4 |  |  |  |  |  |
| bridge-r2 | 8 | 57.2 |  |  |  |  |  |
| bridge-r3 | 1 | 115.2 |  |  |  |  |  |
| bridge-r3 | 2 | 75.4 |  |  |  |  |  |
| bridge-r3 | 4 | 61.0 |  |  |  |  |  |
| bridge-r3 | 8 | 57.0 |  |  |  |  |  |
| bridge-rc-r2 | 1 | 173.1 |  |  |  |  |  |
| bridge-rc-r2 | 2 | 100.5 |  |  |  |  |  |
| bridge-rc-r2 | 4 | 80.3 |  |  |  |  |  |
| bridge-rc-r2 | 8 | 74.6 |  |  |  |  |  |
| bridge-rc-r3 | 1 | 175.6 |  |  |  |  |  |
| bridge-rc-r3 | 2 | 106.7 |  |  |  |  |  |
| bridge-rc-r3 | 4 | 79.9 |  |  |  |  |  |
| bridge-rc-r3 | 8 | 74.8 |  |  |  |  |  |
| bridge-rc | 1 | 172.3 | 73.8 | 43% | 4,068 | 293 | 272.7 |
| bridge-rc | 2 | 109.4 | 69.2 | 63% | 2,049 | 154 | 141.0 |
| bridge-rc | 4 | 81.7 | 66.1 | 81% | 1,034 | 84 | 86.3 |
| bridge-rc | 8 | 75.1 | 64.5 | 86% | 526 | 49 | 72.1 |
| hf-sdpa-rdsp-35b | 1 | 71.0 |  |  |  |  |  |
| hf-sdpa-rdsp-35b | 8 | 50.5 |  |  |  |  |  |
| hf-sdpa-rdsp-r2 | 1 | 70.0 |  |  |  |  |  |
| hf-sdpa-rdsp-r2 | 2 | 57.2 |  |  |  |  |  |
| hf-sdpa-rdsp-r2 | 4 | 52.5 |  |  |  |  |  |
| hf-sdpa-rdsp-r2 | 8 | 50.4 |  |  |  |  |  |
| hf-sdpa-rdsp-r3 | 1 | 68.7 |  |  |  |  |  |
| hf-sdpa-rdsp-r3 | 2 | 57.1 |  |  |  |  |  |
| hf-sdpa-rdsp-r3 | 4 | 52.6 |  |  |  |  |  |
| hf-sdpa-rdsp-r3 | 8 | 50.5 |  |  |  |  |  |
| hf-sdpa-rdsp-rc-r2 | 1 | 114.8 |  |  |  |  |  |
| hf-sdpa-rdsp-rc-r2 | 2 | 77.9 |  |  |  |  |  |
| hf-sdpa-rdsp-rc-r2 | 4 | 68.0 |  |  |  |  |  |
| hf-sdpa-rdsp-rc-r2 | 8 | 64.4 |  |  |  |  |  |
| hf-sdpa-rdsp-rc-r3 | 1 | 114.2 |  |  |  |  |  |
| hf-sdpa-rdsp-rc-r3 | 2 | 79.6 |  |  |  |  |  |
| hf-sdpa-rdsp-rc-r3 | 4 | 68.5 |  |  |  |  |  |
| hf-sdpa-rdsp-rc-r3 | 8 | 64.9 |  |  |  |  |  |
| hf-sdpa-rdsp-rc | 1 | 113.3 | 62.4 | 55% | 3,088 | 175 | 194.4 |
| hf-sdpa-rdsp-rc | 2 | 80.9 | 60.4 | 75% | 1,646 | 88 | 105.4 |
| hf-sdpa-rdsp-rc | 4 | 68.9 | 58.7 | 85% | 893 | 44 | 72.7 |
| hf-sdpa-rdsp-rc | 8 | 64.6 | 57.5 | 89% | 517 | 22 | 84.1 |
| hf-varlen-rdsp | 1 | 80.5 | 49.4 | 61% | 2,467 | 94 | 122.2 |
| hf-varlen-rdsp | 2 | 59.0 | 47.7 | 81% | 1,308 | 47 | 68.7 |
| hf-varlen-rdsp | 4 | 52.9 | 46.3 | 87% | 711 | 24 | 51.9 |
| hf-varlen-rdsp | 8 | 51.0 | 45.4 | 89% | 412 | 12 | 48.5 |

## fwd, 1 image per call: kernels per image, by part

| Part | bridge-35b | bridge-cg-n1-r2 | bridge-cg-n1-r3 | bridge-cg-n1 | bridge-cg-n2 | bridge-cg-n4 | bridge-cg-n8 | bridge-r2 | bridge-r3 | bridge-rc-r2 | bridge-rc-r3 | bridge-rc | hf-sdpa-rdsp-35b | hf-sdpa-rdsp-r2 | hf-sdpa-rdsp-r3 | hf-sdpa-rdsp-rc-r2 | hf-sdpa-rdsp-rc-r3 | hf-sdpa-rdsp-rc | hf-varlen-rdsp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rotary |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 378 |
| core_attn |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 27 |
| linear |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 108 |
| norm |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 54 |
| attn_other |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 27 |
| mlp_other |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 27 |
| layer_other |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 54 |
| outside |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 107 |

## fwd, 1 image per call: GPU ms per image, by part

| Part | bridge-35b | bridge-cg-n1-r2 | bridge-cg-n1-r3 | bridge-cg-n1 | bridge-cg-n2 | bridge-cg-n4 | bridge-cg-n8 | bridge-r2 | bridge-r3 | bridge-rc-r2 | bridge-rc-r3 | bridge-rc | hf-sdpa-rdsp-35b | hf-sdpa-rdsp-r2 | hf-sdpa-rdsp-r3 | hf-sdpa-rdsp-rc-r2 | hf-sdpa-rdsp-rc-r3 | hf-sdpa-rdsp-rc | hf-varlen-rdsp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rotary |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 4.57 |
| core_attn |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 4.31 |
| linear |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3.79 |
| norm |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.58 |
| attn_other |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.04 |
| mlp_other |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.53 |
| layer_other |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.34 |
| outside |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1.27 |

## fwd, 1 image per call: CPU ms (traced, forward thread, exclusive) per image, by part

| Part | bridge-35b | bridge-cg-n1-r2 | bridge-cg-n1-r3 | bridge-cg-n1 | bridge-cg-n2 | bridge-cg-n4 | bridge-cg-n8 | bridge-r2 | bridge-r3 | bridge-rc-r2 | bridge-rc-r3 | bridge-rc | hf-sdpa-rdsp-35b | hf-sdpa-rdsp-r2 | hf-sdpa-rdsp-r3 | hf-sdpa-rdsp-rc-r2 | hf-sdpa-rdsp-rc-r3 | hf-sdpa-rdsp-rc | hf-varlen-rdsp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rotary |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 11.40 |
| core_attn |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3.67 |
| linear |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 8.87 |
| norm |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3.85 |
| attn_other |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 11.53 |
| mlp_other |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2.05 |
| layer_other |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3.82 |
| outside |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 4.60 |
