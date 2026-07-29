# Reasoning-Aware Compression (RAC)

One-shot pruning of reasoning LLMs, calibrated on the model's **own chain-of-thought**.

Implements [*Reasoning Models Can be Accurately Pruned Via Chain-of-Thought
Reconstruction*](https://arxiv.org/abs/2509.12464) (Lucas, Behdin, Wang, Tang, Song,
Mazumder — ICLR 2026). Reference code from the authors:
[RyanLucas3/Reasoning-Aware-Compression](https://github.com/RyanLucas3/Reasoning-Aware-Compression).

## Why

Layer-wise pruning methods (SparseGPT, Wanda, ALPS) minimise a reconstruction error

```
min ||W X - W' X||_F^2   s.t.  ||W'||_0 <= S
```

over calibration activations `X`, which are traditionally collected by running **prompts**
(C4 documents, task inputs) through the dense model. That is a reasonable proxy when
inference is prompt-dominated. Reasoning models are the opposite: a MATH-500 problem is a
few hundred tokens and the chain-of-thought that answers it is thousands. Pruning
therefore optimises the wrong distribution, and the failure is worse than a plain accuracy
drop — the pruned model rambles, so it gets **slower** as it gets sparser. In the paper,
DeepSeek-R1-Distill-Qwen-7B pruned to 50% with C4 calibration takes 135 min on MATH-500
versus 23 min dense, while accuracy falls from 0.936 to 0.744.

RAC changes the calibration set and nothing else: it collects the dense model's on-policy
CoT traces and concatenates their activations with the prompt activations,

```
X_RAC = [ X_prompt | X_decode ]
```

so each layer is reconstructed against the activations it will really see while decoding.
It is a drop-in change to any layer-wise pruner, needs no retraining or distillation, and
recovers most of the loss: the same 7B model at 50% sparsity reaches 0.900 accuracy in
35 min.

## How it works

```
        phase I: collect_traces.py                      phase II: prune.py
 ┌────────────────────────────────────┐      ┌──────────────────────────────────────┐
 │ task prompts (math / code)         │      │ calibration windows                  │
 │            │                       │      │  c4     | prompt | rac               │
 │            ▼                       │      │            │                         │
 │ dense reasoning model              │      │            ▼                         │
 │  vLLM | HF | ZeRO-Inference        │      │ block 0 → prune → propagate          │
 │            │                       │      │ block 1 → prune → propagate          │
 │            ▼                       │      │   ...    (one block on GPU at a time)│
 │ prompt + on-policy CoT  ──── jsonl ├─────▶│ block L → prune                      │
 └────────────────────────────────────┘      │            │                         │
                                             │            ▼  sparse checkpoint      │
                                             └──────────────────────────────────────┘
```

Teacher-forcing a sampled trace closely approximates the hidden states the model computed
while generating it, so phase II recovers the decode-time activations of Algorithm 1 in
the paper without re-running generation during pruning. It is an approximation rather than
an identity because traces are stored as decoded text and re-tokenized when the
calibration windows are packed, so a few tokens near window boundaries can differ from
those originally sampled — negligible over a 1M-token calibration set.

## Layout

```
compression/reasoning_aware_compression/
├── collect_traces.py                    # phase I: on-policy CoT rollouts -> jsonl
├── prune.py                             # phase II: one-shot pruning -> sparse checkpoint
├── rac/
│   ├── calibration.py                   # c4 / prompt / rac calibration sets  <- the method
│   ├── sequential.py                    # block-by-block calibration driver
│   ├── sparsegpt.py                     # SparseGPT Hessian + reconstruction
│   ├── wanda.py                         # Wanda activation-norm pruner
│   ├── magnitude.py                     # magnitude baseline (calibration-free)
│   └── modelutils.py                    # layer discovery, sparsity reporting
├── bash_script/
│   ├── run_rac_qwen1_5b.sh              # end-to-end: traces -> prune -> eval
│   ├── run_calibration_ablation.sh      # c4 vs prompt vs rac, same budget
│   ├── collect_traces_zero_inference.sh # traces for 32B/70B via ZeRO-Inference
│   └── eval_math500.sh                  # lighteval MATH-500 (accuracy + runtime)
├── tests/                               # CPU-only unit tests (pytest)
├── third_party_licenses/                # upstream licenses for the adapted pruners
└── requirements.txt
```

## Quick start

```bash
cd compression/reasoning_aware_compression
pip install -r requirements.txt
pip install vllm          # optional but much faster for phase I
```

**Phase I — collect on-policy traces** (paper: 1M tokens, `T_max = 8192`):

```bash
python collect_traces.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --dataset open-r1/OpenR1-Math-220k --prompt-column problem \
    --trace-tokens 1_000_000 --max-new-tokens 8192 \
    --output outputs/traces/r1-qwen-1.5b_math.jsonl
```

**Phase II — prune against them**:

```bash
python prune.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --calibration rac --traces outputs/traces/r1-qwen-1.5b_math.jsonl \
    --sparsity 0.5 --nsamples 512 --seqlen 2048 \
    --output outputs/checkpoints/r1-qwen-1.5b-rac-50
```

**Evaluate** (accuracy *and* runtime — both matter here):

```bash
MODEL_DIR=outputs/checkpoints/r1-qwen-1.5b-rac-50 bash bash_script/eval_math500.sh
```

Or run all three phases at once with `bash bash_script/run_rac_qwen1_5b.sh`.

### Calibration sets

`--calibration` selects the ablation axis of the paper. Everything else stays fixed.

| Flag | Calibration tokens | Notes |
| --- | --- | --- |
| `--calibration c4` | generic web text | SparseGPT/Wanda default; `--dataset allenai/c4 --dataset-config en` |
| `--calibration prompt` | task prompts only | pass `--traces` to reuse exactly the prompts RAC saw, or `--dataset` for a fresh set |
| `--calibration rac` | prompts **+** on-policy CoT | the method; requires `--traces` |

`--nsamples × --seqlen` is the token budget (`512 × 2048 = 1M`, as in the paper).
Documents are packed into fixed-length windows, so no padding tokens enter the Hessian.

### Other knobs

| Flag | Meaning |
| --- | --- |
| `--pruning-method` | `sparsegpt` (default), `wanda`, `magnitude`. RAC helps all calibrated methods (Table 9). |
| `--prune-n 2 --prune-m 4` | semi-structured 2:4 sparsity for Ampere+ sparse tensor cores |
| `--scope mlp` | prune only feed-forward projections, leave attention dense |
| `--layer-thirds 1,3` | prune only the first and last third of the decoder stack (Table 6) |
| `--batch-size` | calibration samples per forward pass; raise it if the GPU has headroom |

## Scaling to large models

**Pruning** streams the model through the accelerator one transformer block at a time —
the block, its Hessians and one activation slice are the only device-resident tensors —
so a 70B checkpoint prunes on a single 80GB GPU, as in the paper. Nothing needs to be
sharded and `prune.py` is a single-process script. Host memory holds the weights plus two
activation buffers of `nsamples × seqlen × hidden_size` (~17GB each at the default budget
for a 70B model in bf16); drop `--nsamples` if that is tight.

**Trace collection** is the expensive phase and is where DeepSpeed helps. For models that
do not fit in aggregate GPU memory, `--backend deepspeed` runs generation under
ZeRO-Inference (ZeRO-3 parameter sharding with CPU or NVMe offload), following
[`inference/huggingface/zero_inference`](../../inference/huggingface/zero_inference):

```bash
NUM_GPUS=2 MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    bash bash_script/collect_traces_zero_inference.sh
```

Use `NVME_OFFLOAD_DIR=/path/on/nvme` to spill parameters to NVMe instead of CPU memory.
When the model does fit, `--backend vllm` is considerably faster.

The pruned checkpoint is a standard Hugging Face directory, so it serves through
DeepSpeed-Inference, vLLM or plain `transformers` unchanged. Unstructured sparsity gives
memory and (with a sparse kernel) compute savings; 2:4 checkpoints run on Ampere+ sparse
tensor cores — the paper measures 1561 tok/s at 2:4 versus 1426 tok/s dense for
DeepSeek-R1-Distill-14B.

## Results from the paper

MATH-500 acc@1 and total evaluation time, one-shot SparseGPT, 1M calibration tokens,
32k decode budget. These are the authors' published numbers (Tables 1 and 2), reproduced
here for reference — this example is the pipeline that generates them, not a re-run.

| Model | Sparsity | C4 | Prompt-only | **RAC** | Runtime C4 → RAC |
| --- | --- | --- | --- | --- | --- |
| R1-Distill-Qwen-1.5B (dense 0.832) | 40% | 0.658 | 0.728 | **0.774** | 58 → 32 min |
| | 50% | 0.356 | 0.496 | **0.664** | 157 → 57 min |
| R1-Distill-Qwen-7B (dense 0.936) | 40% | 0.890 | 0.898 | **0.912** | 38 → 29 min |
| | 50% | 0.744 | 0.812 | **0.900** | 135 → 35 min |
| Qwen3-8B (dense 0.962) | 40% | 0.906 | 0.944 | **0.968** | 58 → 29 min |
| | 50% | 0.564 | 0.470 | **0.862** | 259 → 17 min |

The gap widens with sparsity, and the runtime column is the practical point: the C4-
calibrated models are both less accurate and slower than the dense baseline. The paper
reports the same pattern on AIME-25 and LiveCodeBench, and finds that on-policy traces
(from the model being pruned) beat off-policy traces from a larger model.

## Tests

CPU-only, no network, a few seconds:

```bash
cd compression/reasoning_aware_compression
pytest tests
```

They cover calibration-window packing, the RAC-vs-prompt distinction, target sparsity for
SparseGPT/Wanda/magnitude, 2:4 patterns, block and scope selection, batched-equals-
unbatched calibration on a tiny randomly-initialised Qwen2, and the failure paths
(unsupported architectures, config restored after a failed run).

## Attribution

- `rac/sparsegpt.py` adapts [IST-DASLab/sparsegpt](https://github.com/IST-DASLab/sparsegpt)
  (Apache-2.0); upstream license text in
  [`third_party_licenses/LICENSE.sparsegpt`](third_party_licenses/LICENSE.sparsegpt).
- `rac/wanda.py` adapts [locuslab/wanda](https://github.com/locuslab/wanda)
  (MIT, © 2023 CMU Locus Lab); upstream license text in
  [`third_party_licenses/LICENSE.wanda`](third_party_licenses/LICENSE.wanda).
- The RAC method and its reference implementation are by the paper's authors
  ([repo](https://github.com/RyanLucas3/Reasoning-Aware-Compression)); this example is a
  standalone reimplementation for DeepSpeedExamples.

```bibtex
@inproceedings{lucas2026rac,
  title     = {Reasoning Models Can be Accurately Pruned Via Chain-of-Thought Reconstruction},
  author    = {Lucas, Ryan and Behdin, Kayhan and Wang, Zhipeng and Tang, Shao and
               Song, Qingquan and Mazumder, Rahul},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2509.12464}
}
```
