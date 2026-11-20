# Subchannel FP8/INT8 Quantized Training Example

DeepSpeed's `quantization.subchannel` config section runs the matrix multiplications of `nn.Linear` layers and AutoEP grouped experts in FP8 or INT8 instead of BF16, with no model code changes.
This example continues training a pretrained causal LM on WikiText-103 under several quantization recipes, so that validation loss, step time and peak memory can be compared with a BF16 baseline.
A second script, `from_scratch.py`, trains a model from random initialization to convergence to compare final quality.

The feature is proposed in [DeepSpeed issue #9132](https://github.com/deepspeedai/DeepSpeed/issues/9132) and implemented in [PR #9131](https://github.com/deepspeedai/DeepSpeed/pull/9131), which has not been merged. See the tutorial in that PR (`docs/_tutorials/subchannel-quantized-training.md`) for what each setting does.

## Prerequisites

- An NVIDIA GPU of compute capability 8.9 or newer (for example L40S, H100, H200) for the FP8 kernels. On other devices the GEMMs fall back to a PyTorch reference implementation that reproduces the numerics but is not meant for speed.
- `pip install -r requirements.txt`, which installs DeepSpeed from the tip of PR #9131.
- For MoE models, a GPU count that divides the model's expert count (the script sets `autoep_size` to the GPU count).

## Recipes

`--recipe` selects what goes into the `quantization` section of the DeepSpeed config:

| Recipe | Format | Scaling | Purpose |
| ------ | ------ | ------- | ------- |
| `bf16` | none | none | Baseline: no `quantization` section. |
| `e4m3_row` | FP8 e4m3 | one scale per operand row (`block_size: 0`) | The fast mode. Faster than BF16 on wide layers only. |
| `e4m3` | FP8 e4m3 | 128-element blocks | The feature's defaults. Closer to BF16 in loss, slower than BF16. |
| `int8` | INT8 | 128-element blocks | INT8 operands. |
| `int8_h128` | INT8 | 128-element blocks, Hadamard rotation of 128 | INT8 with outliers spread across each block. |

All quantized recipes round to nearest in the forward GEMM and use stochastic rounding in the two gradient GEMMs. For example, `e4m3_row` produces:

```json
"quantization": {
  "subchannel": {
    "enabled": true,
    "forward": {"block_size": 0, "stochastic_rounding": false},
    "dx": {"block_size": 0, "stochastic_rounding": true},
    "dw": {"block_size": 0, "stochastic_rounding": true}
  }
}
```

## Run

Launch one process per GPU with `torchrun`. Run the `bf16` recipe first, then any quantized recipe with the same arguments.

A dense model with wide layers, where row-scaled FP8 is faster than BF16:

```bash
for recipe in bf16 e4m3_row e4m3; do
  torchrun --nproc_per_node 4 train.py --recipe $recipe \
      --model Qwen/Qwen2.5-14B --dense --grad_checkpointing --bf16_states \
      --steps 100 --eval_every 50 --seq_len 2048 --batch_size 4 --eval_blocks 32 \
      --out results/qwen2.5-14b_$recipe.json
done
```

An MoE model with AutoEP, where the quantized recipes match BF16 in loss but are slower:

```bash
for recipe in bf16 e4m3_row e4m3 int8 int8_h128; do
  torchrun --nproc_per_node 4 train.py --recipe $recipe \
      --model Qwen/Qwen1.5-MoE-A2.7B \
      --out results/qwen1.5-moe_$recipe.json
done
```

Each run prints validation loss at every `--eval_every` steps and, at the end, the median step time and peak GPU memory. The same numbers are written to the `--out` JSON file together with the per-step training loss and step times.

Options:

- `--dense`: the model is not an MoE model; train with plain data parallelism instead of AutoEP.
- `--grad_checkpointing`: enable gradient checkpointing. Needed to fit wide dense models at thousands of tokens per step.
- `--bf16_states`: keep master weights and optimizer state in BF16. Needed to fit models of roughly 14B parameters and up on 141 GiB GPUs.
- `--zero_stage`: ZeRO stage (default 2). Use 3 when parameters plus gradients do not fit on one GPU.
- `--steps`, `--eval_every`, `--seq_len`, `--batch_size`, `--eval_blocks`, `--lr`: training and evaluation sizes.

The script reads at most the first 60,000 rows of WikiText-103, about enough for 100 steps at 4 GPUs x batch 4 x sequence 2048. Lower `--steps` or the batch shape if you raise any of these.

## Full training from scratch

`from_scratch.py` trains a Llama-architecture model built from a config (default: 8 layers, hidden 4096, MLP 11008, 1.82B parameters, GPT-2 tokenizer) on all of WikiText-103 with a warmup + cosine schedule that decays to zero, and reports final validation and test loss.
It supports the `bf16`, `e4m3_row` and `e4m3` recipes and plain data parallelism under ZeRO-2.

Tokenize the dataset once, in a single process:

```bash
python from_scratch.py --prep
```

Then train each recipe with the same `--seed`, which fixes both the initial weights and the data order:

```bash
for seed in 0 1; do
  for recipe in bf16 e4m3_row e4m3; do
    torchrun --nproc_per_node 2 from_scratch.py --recipe $recipe --seed $seed \
        --out results/fs_${recipe}_s${seed}.json
  done
done
```

The defaults are 4 epochs (28,788 steps on 2 GPUs), batch 4 x sequence 2048 per GPU, AdamW with peak lr 2e-4, 1000 warmup steps, weight decay 0.1 and gradient clipping 1.0.
One run takes about 2.5 to 3 hours on 2 H200s and needs about 30 GiB per GPU.
Validation loss is evaluated and the `--out` file rewritten every `--eval_every` steps (default 500).

Options:

- `--seed`, `--epochs`, `--lr`, `--warmup`, `--seq_len`, `--batch_size`, `--eval_every`: training schedule and sizes.
- `--layers`, `--hidden`, `--intermediate`: model shape. The head count is `hidden / 128`.
- `--max_steps`: stop early without changing the schedule, for timing runs.
- `--data_dir`: where `--prep` writes the tokenized splits (default `data/` next to the script).

The number of steps per epoch, and so the schedule, depends on the GPU count.

## Reference results

Measured on 4x NVIDIA H200 (driver 595.112.07) with torch 2.8.0+cu129, Triton 3.4.0 and transformers 5.23.0. The full-training table has two seeds; the others are one run per configuration.

Dense models, with the first command above (Qwen2.5-32B additionally with `--zero_stage 3`):

| Model | Recipe | Val loss @100 | Step p50 | vs BF16 | Peak memory |
| ----- | ------ | ------------- | -------- | ------- | ----------- |
| Qwen2.5-14B | `bf16` | 1.6783 | 2145 ms | 1.00x | 66.2 GiB |
| | `e4m3_row` | 1.6839 | 1965 ms | 1.09x | 66.2 GiB |
| | `e4m3` | 1.6816 | 2896 ms | 0.74x | 66.2 GiB |
| Qwen2.5-32B | `bf16` | 1.6561 | 4594 ms | 1.00x | 82.2 GiB |
| | `e4m3_row` | 1.6604 | 3801 ms | 1.21x | 82.2 GiB |
| | `e4m3` | 1.6584 | 6175 ms | 0.74x | 82.2 GiB |

MoE models, 300 steps at the default batch 4 x sequence 512 (Qwen3-30B-A3B additionally with `--bf16_states`):

| Model | Recipe | Val loss @300 | Step p50 | Peak memory |
| ----- | ------ | ------------- | -------- | ----------- |
| Qwen1.5-MoE-A2.7B | `bf16` | 1.8900 | 341 ms | 64.5 GiB |
| | `e4m3_row` | 1.8941 | 518 ms | 64.5 GiB |
| | `e4m3` | 1.8923 | 500 ms | 64.5 GiB |
| Qwen3-30B-A3B | `bf16` | 1.9084 | 622 ms | 91.8 GiB |
| | `e4m3_row` | 1.9116 | 947 ms | 91.8 GiB |
| | `e4m3` | 1.9118 | 789 ms | 91.8 GiB |

Full training from scratch with the `from_scratch.py` defaults, 2 GPUs per run, two runs sharing the machine at a time:

| Recipe | Seed | Final val loss | Test loss | Test ppl | Step p50 | Peak memory |
| ------ | ---- | -------------- | --------- | -------- | -------- | ----------- |
| `bf16` | 0 | 2.9597 | 2.9761 | 19.61 | 323 ms | 29.9 GiB |
| `e4m3_row` | 0 | 2.9557 | 2.9747 | 19.58 | 307 ms | 29.9 GiB |
| `e4m3` | 0 | 2.9518 | 2.9738 | 19.57 | 386 ms | 29.9 GiB |
| `bf16` | 1 | 2.9650 | 2.9860 | 19.81 | 326 ms | 29.9 GiB |
| `e4m3_row` | 1 | 2.9631 | 2.9828 | 19.74 | 305 ms | 29.9 GiB |
| `e4m3` | 1 | 2.9545 | 2.9792 | 19.67 | 386 ms | 29.9 GiB |

What to expect:

- In full training, both FP8 recipes end within seed-to-seed variation of BF16: the two BF16 seeds differ by 0.010 in test loss, and each FP8 run is within 0.007 of BF16 at the same seed. This covers one dense architecture and one dataset; MoE models and the INT8 recipes have not been trained to convergence.

- Row-scaled FP8 beats BF16 only when layers are wide (about 5000 and up) and each GPU processes several thousand tokens per step. On the MoE models above, whose layers are about 2048 wide, it is about 1.5x slower.
- Block-scaled recipes are slower than BF16 on every model measured.
- Peak memory does not change: only the GEMM operands are quantized, on the fly.
- The `train.py` runs are short continued pretraining. They show that loss tracks the baseline over a few hundred steps, not final model quality.
