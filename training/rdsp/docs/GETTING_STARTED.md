# Getting started

Train a model with rdsp on one multi-GPU node.

## Install

Linux, CUDA GPUs, Python 3.12. `bench/setup_node.sh` does all of this on a fresh node. By hand:

```bash
uv venv -p 3.12 && . .venv/bin/activate
uv pip install "torch==2.13.0" "torchvision==0.28.0" "transformers==5.17.0" "ray==2.59.0" cloudpickle \
    "accelerate==1.15.0" safetensors datasets pillow
DS_BUILD_OPS=0 uv pip install --no-build-isolation \
    "git+https://github.com/deepspeedai/DeepSpeed.git@53a2ac44fb664bea838df3981ba4366b91643070"
uv pip install -e .
uv pip install "flash-linear-attention==0.5.2" "liger-kernel==0.8.4"   # Qwen3.5 linear attention, fused kernels
uv pip install --no-build-isolation "causal-conv1d==1.7.0"
```

The DeepSpeed commit is the one every result was measured with; the other versions are the ones the October runs used.

## Terms

| Term | Meaning |
|---|---|
| Stage | A contiguous slice of the model on its own group of GPUs. |
| PP | Pipeline parallelism: the number of stages. |
| TP | Tensor parallelism: each layer's matrices split across GPUs. |
| DP | Data parallelism: copies of a stage, each on different rows. ZeRO-1 splits the optimizer state across them. |
| Cut | The decoder layer where one stage ends and the next begins. |
| Microbatch | The rows one stage processes at a time. A step is many microbatches, with gradients accumulated across them. |
| TP2×PP2×DP2 | 2 stages, each 4 GPUs as TP2 × DP2: 8 GPUs. |

## Run a recipe

`recipes/` has one script per validated layout:

| Recipe | Model | Layout | GPUs |
|---|---|---|---|
| `qwen35-2b-vision1-tp2dp2.sh` | Qwen3.5-2B | vision on 1 GPU + language TP2×DP2 | 5 |
| `qwen35-2b-tp2pp2dp2.sh` | Qwen3.5-2B | TP2×PP2×DP2 | 8 |
| `qwen35-4b-tp2pp2dp2.sh` | Qwen3.5-4B | TP2×PP2×DP2 | 8 |
| `qwen35-4b-tp4pp2dp1.sh` | Qwen3.5-4B | TP4×PP2×DP1 | 8 |
| `qwen35-4b-vision1-tp2dp2.sh` | Qwen3.5-4B | vision on 1 GPU + language TP2×DP2 (80 GB GPUs) | 5 |
| `qwen35-4b-coloc-tp2pp2dp2.sh` | Qwen3.5-4B | vision on every GPU + language TP2×PP2×DP2 (80 GB GPUs) | 8 |
| `qwen3-text-pp.sh` | Qwen3-0.6B | DP2 (ZeRO-2) → TP2 | 4 |

```bash
./recipes/qwen35-2b-tp2pp2dp2.sh
./recipes/qwen35-2b-tp2pp2dp2.sh --steps 200 --lr 2e-5   # extra flags override the recipe
```

Each step prints one line:

```
step 12 loss 0.0931 5312 ms 12187 real tok/s 24672 padded tok/s real 64725 supervised 12304
```

`real` counts tokens that aren't padding; `supervised` counts the tokens the loss is computed on. The first 2–5 steps include compilation and are slower.

## Change the model, data or layout

**Model.** `--model` takes a Hugging Face id. Vision-language models go through `train_vl.py`, text models through `train.py`. Each stage loads only its own weights.

**Data.** `train_vl.py --dataset`:

- `cord-v2`: downloads CORD-v2, scales images to at most `--max-pixels`, and pads rows to `--seq` (with `--pad-multiple N --pad-per-microbatch`, each microbatch to its longest row, as the recipes do).
- `exported:DIR`: steps exported from Megatron-Bridge's loader (`bench/mimo/runs.sh export`). With `--pad-multiple 128 --pad-per-microbatch`, each microbatch is padded to its longest row.
- `synthetic`: random images, for smoke tests.

For your own data, write a loader that yields `(inputs, labels)` entries (see `docs/CONTRACTS.md`, Data) and call `rdsp.initialize(...)` directly. `train_vl.py` is a complete example.

**Layout.** Flags:

- `--stages N`: pipeline stages.
- `--stage S:gpus=G,tp=T,zero=Z`: stage S's GPU count, tensor parallelism and ZeRO stage. The data-parallel degree is G / T. Other keys: `sp`, `ep`, `recompute=1`, `recompute_blocks=N` (recompute only the stage's first N decoder layers, plus its vision encoder; under `--cuts measured`, N moves with the layers the stage gains or loses, so the layers that keep their activations stay as many), `offload=1`, `compile=1`, `compile_vision=1`.
- `--cuts balanced` (default): rdsp estimates where to cut from the model and the first batch. `--cuts measured` starts there, times every stage over `--calibration-steps` steps at learning rate 0, and re-cuts: each stage's time beyond its layers' (vision encoder, embeddings, output layer and loss, kernel launches) counts as a fixed cost of that stage, and no stage gains layers past its GPU's memory. The run then starts from the loaded weights and the first batch. `--cuts even` splits the decoder layers evenly; `--cuts 5,18` sets them by hand. A first cut of `0` puts the vision encoder alone on stage 0: `--cuts 0,balanced` cuts the decoder over the other stages by estimate, `--cuts 0,measured` then re-cuts the decoder stages as `measured` does (the vision stage keeps its GPUs).
- `--master-dtype remainder`: exact fp32 master weights stored as the bf16 weights plus a 16-bit remainder, 2 bytes per parameter instead of 4 (Megatron's `store_param_remainders`). Needs `--grad-dtype bf16` and ZeRO 1 or 2. Works with expert parallelism (`ep=`): the expert weights keep their own group.
- `--moe-experts hf` (mixture-of-experts models, with `--liger`): the routed experts run Hugging Face's grouped matmuls instead of Liger's fused kernel (`liger`, the default). Which is faster depends on the model and GPU; time a few steps of each.
- `--vision-attn varlen`: the vision encoder's attention as one variable-length FlashAttention call per layer (PyTorch's `varlen_attn`), instead of one call per image and three host syncs per layer under SDPA. CUDA only.
- `--colocated-vision`: run the vision encoder on every GPU instead of on stage 0.

`docs/CONTRACTS.md` lists the validated layouts and what isn't supported.

## Common failures

**Not enough GPUs.** The layout asks for more GPUs than the node has:

```
ValidationError: the pipeline needs 8 GPUs; this Ray cluster has 4. Reduce gpus= in the stage layouts, or add GPUs
```

On a multi-node cluster, stages fill the driver's node first, then the next. A stage with more GPUs than a node has spans nodes, its tensor-parallel groups kept inside a node (each node's share of the stage must be a multiple of `tp` x `sp`). A stage that cannot be placed at all fails after `RDSP_PLACEMENT_TIMEOUT_S` (default 600 s) with `stage S: cannot place G GPUs ... in this Ray cluster`.

Two more timeouts, both in seconds:
- `RDSP_P2P_TIMEOUT_S` (default 300): how long a stage waits on a neighbour's send or receive before the step fails.
- `RDSP_ALIVE_TIMEOUT_S` (default 60): how long loading a checkpoint waits for every stage actor to answer a ping. A stage that doesn't answer counts as lost: its workers are rebuilt, or the load fails if the pipeline can't rebuild them.

**Several nodes.** Start Ray on each node yourself (`ray start --head` on the first, `ray start --address <first node>:6379` on the others) with the network settings NCCL needs in that shell, and run `train_vl.py` on the first node with `RAY_ADDRESS=auto`. Workers of a cluster started this way run in `ray start`'s environment, not the driver's; `train_vl.py` passes them the driver's `RDSP_*`, `NCCL_*`, `FI_*`, `HF_*`, `PYTORCH_*`, `TORCH_*`, `DS_*`, `CUDA_HOME`, `LD_LIBRARY_PATH` and `TRANSFORMERS_OFFLINE`, except anything named like a credential (`HF_TOKEN`): the workers read the weights the driver downloaded. On AWS with EFA: `bench/two_node/` (`efa_env.sh`, `ray_cluster.sh`).

**TP doesn't divide the key/value heads.** Qwen3.5-2B has 2 key/value heads:

```
ValidationError: stage 0: tp=4 does not divide the model's 2 key/value heads; use a tp that divides 2
```

**Out of memory.** `torch.OutOfMemoryError` on the first or second step, in order of cost:

1. `--master-dtype remainder` (with `--grad-dtype bf16`): 2 bytes less per parameter of master weights, split over the data-parallel GPUs. The master weights are still exact fp32.
2. Move layers off the full stage: stage 0 holds the vision encoder plus the most microbatches in flight, so try a lower first cut, e.g. `--cuts 4`.
3. Add data parallelism so ZeRO splits the optimizer state over more GPUs.
4. `recompute=1` on the full stage. This keeps only each block's input and recomputes the rest, about 30% more compute on that stage. `recompute_blocks=N` recomputes only the first N layers, for a stage that almost fits.

**`--drop-padding-mask needs rows padded on the right only`.** Your data has padding before real tokens. Drop the flag.
