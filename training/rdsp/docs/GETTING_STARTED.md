# Getting started

Install rdsp, train with a recipe, and rerun the published benchmarks.

## Install

Linux, CUDA GPUs, Python 3.12. On a fresh node:

```bash
bash bench/setup_node.sh
```

It builds `~/venv` with torch 2.13.0, Transformers 5.17.0, Ray 2.59.0, DeepSpeed at commit 53a2ac4 (every result was measured with it), rdsp and the Qwen3.5 kernels, and pulls the NeMo container the Megatron comparisons run in. To install by hand, run its `uv pip install` lines in your own environment. DeepSpeed 0.19.3 is the first PyPI release that includes that commit (`pip install "deepspeed>=0.19.3"`).

## Run a recipe

Terms: a *stage* is a contiguous slice of the model on its own GPUs, and a *cut* is the layer where one stage ends. PP is the number of stages; TP splits each layer's matrices across GPUs; DP gives copies of a stage different rows, with ZeRO splitting their optimizer state; EP splits a mixture-of-experts layer's experts. A *microbatch* is the rows a stage processes at a time. TP2×PP2×DP2 is 2 stages of 4 GPUs, each TP2 × DP2.

Each script in `recipes/` is one layout with a published result, plus a text-only starting point. The first 2–5 steps compile and are slower.

| Recipe | Layout | GPUs |
|---|---|---|
| `qwen35-2b-vision1-tp2dp2.sh` | vision encoder on 1 GPU + language TP2×DP2 | 5 |
| `qwen35-2b-tp2pp2dp2.sh` | 2 stages, each TP2×DP2 | 8 |
| `qwen35-4b-tp2pp2dp2.sh` | 2 stages, each TP2×DP2 | 8 |
| `qwen35-4b-tp4pp2dp1.sh` | 2 stages, each TP4 | 8 |
| `qwen35-4b-vision1-tp2dp2.sh` | vision encoder on 1 GPU + language TP2×DP2 | 5, 80 GB each |
| `qwen35-4b-coloc-tp2pp2dp2.sh` | vision encoder on every GPU + 2 stages of TP2×DP2 | 8, 80 GB each |
| `qwen3-text-pp.sh` | Qwen3-0.6B: stage 0 DP2 (ZeRO-2), stage 1 TP2 | 4 |

```bash
./recipes/qwen35-2b-tp2pp2dp2.sh
./recipes/qwen35-2b-tp2pp2dp2.sh --steps 200 --lr 2e-5   # extra flags override the recipe
```

## Change the model, data or layout

**Model.** `--model` takes a Hugging Face id: `train_vl.py` for vision-language models, `train.py` for text models. Each stage loads only its own weights.

**Data.** `train_vl.py --dataset` takes `cord-v2` (images scaled to at most `--max-pixels`), `exported:DIR` (rows exported from Megatron-Bridge's loader by `bench/common/runs.sh export`, as the benchmarks use) or `synthetic`. With `--pad-multiple N --pad-per-microbatch`, as in the recipes, each microbatch is padded to its longest row instead of `--seq`. For your own data, pass `engine.train_batch` an iterator of `(inputs, labels)` entries, as `train_vl.py` does: inputs are a tensor or a dict of tensors (vision models add `pixel_values` and `image_grid_thw`), and each call reads exactly one step's microbatches.

**Layout.** Validated layouts are listed in `src/ray_deepspeed_pipeline/support_matrix.py`.

- `--stages N --stage S:gpus=G,tp=T,zero=Z`: N stages; stage S's GPU count, tensor parallelism and ZeRO stage, with data parallelism G / T. Other keys: `sp`, `ep`, `fold`, `recompute=1`, `recompute_blocks=N` (only the stage's first N decoder layers, plus its vision encoder), `offload=1`, `compile=1`, `compile_vision=1`.
- `--cuts balanced` (default) estimates the cuts from the model and the first batch. `--cuts measured` then times every stage for `--calibration-steps` steps at learning rate 0 and re-cuts within each GPU's memory. `--cuts even` splits evenly; `--cuts 5,18` sets the cuts by hand.
- A first cut of `0` (`--cuts 0`, `0,balanced` or `0,measured`) gives the vision encoder a stage of its own. `--colocated-vision` runs it on every GPU instead.
- `--master-dtype remainder` (with `--grad-dtype bf16`, ZeRO 1 or 2) stores the exact fp32 master weights as the bf16 weights plus a 16-bit remainder: 2 bytes per parameter instead of 4.

**Out of memory.** Cheapest fix first: `--master-dtype remainder --grad-dtype bf16`; a lower first cut, such as `--cuts 4`, since stage 0 holds the vision encoder and the most microbatches in flight; more data parallelism; `recompute=1` on the full stage (about 30% more compute there), or `recompute_blocks=N`.

## Environment variables

| Variable | Effect |
|---|---|
| `RDSP_PLACEMENT_TIMEOUT_S` (600) | How long a stage waits for its GPUs |
| `RDSP_P2P_TIMEOUT_S` (300) | How long a stage waits on a neighbour's send or receive before the step fails |
| `RDSP_ALIVE_TIMEOUT_S` (60) | How long `load_checkpoint()` waits for each stage actor; a silent one is rebuilt |
| `RDSP_PROFILE=1` | Per rank and step: forward, backward, waiting and optimizer time (as `train_vl.py --profile`; slows steps) |
| `RDSP_TORCH_PROFILE=<step>` | A torch profiler kernel table for that step in `/tmp/rdsp_torch_profile_stage<s>_rank<r>.txt` |

**Several nodes.** Start Ray on each node with NCCL's network settings in that shell (on AWS with EFA, `bench/cluster/ray_cluster.sh`), then run `train_vl.py` on the first node with `RAY_ADDRESS=auto`; it passes its `RDSP_*`, `NCCL_*` and similar variables to the workers. Stages fill the driver's node first, and a stage wider than a node keeps each tensor-parallel group inside one node.

## Reproducing the blog's results

### What every run shares

| | |
|---|---|
| Software | DeepSpeed 53a2ac4 (rdsp), Megatron-Bridge v0.6.0 in the NeMo 26.08 container (also MegatronMIMO), TorchTitan 3c36a77 on torch 2.16.0.dev20261012, Hugging Face Transformers 5.17.0 |
| Hardware | AWS p5.48xlarge: 8× H100 80GB, NVLink, 32 EFA cards; two nodes in one cluster placement group |
| Data | CORD-v2, every row padded to 2,048 tokens; rdsp and TorchTitan replay the rows Megatron-Bridge's loader exports, in the same order |
| Optimizer | AdamW, learning rate 1e-5 constant, betas 0.9 / 0.95, eps 1e-8; no weight decay, clipping or warm-up |
| Loss | Mean over the step's supervised tokens; in the MIMO comparison, the mean per microbatch, as MIMO trains |
| Precision | rdsp and Bridge: BF16 gradients, FP32 master weights stored as BF16 plus a 16-bit remainder (`bench/common/meg_extra_bf16.txt` for Bridge). TorchTitan: FP32 weights and gradients, reduced in BF16. 9B MIMO comparison: FP32 gradients and master weights on both, as MIMO fixes them |
| Step time | Median over steps 6–50 of one 50-step run (`bench/common/medians.py`) |
| 35B only | Router load-balancing loss and multi-token prediction off on every system; no tokens dropped |

### Setting up the nodes

Use the AWS Deep Learning Base OSS NVIDIA Driver GPU AMI (Ubuntu 22.04) with a 500 GB disk. Put this folder at `~/rdsp` on every node (`ln -s "$PWD" ~/rdsp`) and run everything from there, in bash. Node 1 also needs `~/env2.sh` (`export PRIV1=<node 1 private IP> PRIV2=<node 2 private IP>`) and `~/key2.pem`, an SSH key for `ubuntu` on node 2. Then `bash bench/qwen35_27b/prep.sh` on node 1 sets up both nodes: the environments, the 35B and 27B weights, Megatron-Bridge's checkpoints and exported rows, an NCCL check and the two-node Ray cluster. A rerun skips finished steps. `bench/README.md` lists every script.

### Qwen3.5-35B-A3B, 16 GPUs on two nodes

128 rows a step. On node 1:

```bash
RECOMPUTE=1 K=4 bash bench/qwen35_35b/rdsp_vision_stage.sh   # -> logs/vision_stage/rdsp-k4-rc.log
bash bench/qwen35_35b/rdsp_pp8dp2.sh                          # -> logs/pp8dp2/rdsp.log
bash bench/qwen35_35b/torchtitan.sh fsdp16-ep8-bf16
. bench/common/two_node.sh                                    # Megatron-Bridge: MEG_COMMON, MEG_TP1
FAST="rerun_state_machine.check_for_nan_in_loss=false ddp.check_for_nan_in_grad=false dataset.num_workers=8"
NV="\"model.pipeline_model_parallel_layout='Et*17|t*23L'\" model.cross_entropy_fusion_impl=te"
bash bench/cluster/both.sh env MODEL=Qwen3.5-35B-A3B RECIPE=qwen35_vl_35b_a3b_sft_16gpu_h100_bf16_config \
  GBS=128 STEPS=50 LOGS=$HOME/rdsp/bench/qwen35_35b/logs/pp2ep8 NET=efa GRAD_REDUCE_FP32=false \
  CUSTOM_NAME=bridge-pp2ep8-nv CUSTOM_TP=1 CUSTOM_PP=2 \
  MEGATRON_EXTRA="$MEG_COMMON $MEG_TP1 $FAST model.expert_model_parallel_size=8 model.batch_p2p_comm=false model.batch_p2p_sync=false $NV" \
  bash bench/common/runs.sh custom-megatron
```

| System | Layout | Recompute | Step time | Tokens/s per GPU |
|---|---|---|---|---|
| rdsp | vision encoder on a 4-GPU stage, then 6 stages of 2 GPUs; 32 microbatches of 4 rows | every layer | 6.38 s | 2,567 |
| Megatron-Bridge | TP1 × PP2 × EP8 from NVIDIA's H100 recipe, layers 17 \| 23, no interleaving; 16 microbatches of 8 rows | none | 8.67 s | 1,890 |
| TorchTitan | FSDP over 16 GPUs + EP8 inside each node; 8 microbatches of 16 rows | none | 9.11 s | 1,798 |
| rdsp | PP8 × DP2; 64 microbatches of 2 rows | vision encoder only | 9.60 s | 1,707 |

Both rdsp runs measure their own cuts. The vision stage's width, 4, comes from the PP8 × DP2 log (`python3 bench/common/layout_from_log.py vision-width bench/qwen35_35b/logs/pp8dp2/rdsp.log 16`).

### Qwen3.5-27B, 8 GPUs on one node

64 rows a step, on node 1 with the Ray cluster down (`. ~/env2.sh && bash bench/cluster/ray_cluster.sh down $PRIV1 $PRIV2`). rdsp runs eight 1-GPU stages with `--cuts measured`. `ROWS` is the rows per microbatch. `RC_PER` is the layers each stage recomputes, out of 6, 9, 9, 9, 9, 9, 8 and 5 at the calibration cut, so `6,8,7,7,6,5,3,0` keeps the activations of 0, 1, 2, 2, 3, 4, 5 and 5 layers. Without it every layer is recomputed. Megatron-Bridge runs PP8 at rdsp's measured cut (its even split runs out of memory), every layer recomputed; in its layout `E` is the embedding, `t*9` nine layers, `L` the loss. TorchTitan runs on another machine of the same type.

```bash
T=bench/qwen35_27b; mkdir -p $T/logs/pp8
rdsp27() {   # rdsp27 <log name> <settings...>
  ( unset LD_LIBRARY_PATH && . ~/venv/bin/activate && . ~/cuda_env.sh && \
    env "${@:2}" DS_BUILD_OPS=0 HF_HOME=$HOME/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      DATA=exported:$HOME/mimo/cord_steps-Qwen3.5-27B bash $T/rdsp_tune.sh ) > $T/logs/pp8/$1.log 2>&1
}
rdsp27 rdsp-rows1 ROWS=1 RC_PER=6,8,7,7,6,5,3,0
rdsp27 rdsp-rows2 ROWS=2 RC_PER=6,8,7,7,6,5,3,0
rdsp27 rdsp-rows4 ROWS=4 RC_PER=6,9,8,8,8,7,5,4
RC_ALL="model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1"
FAST="rerun_state_machine.check_for_nan_in_loss=false ddp.check_for_nan_in_grad=false dataset.num_workers=8"
bridge27() {  # bridge27 <rows per microbatch> <rdsp's measured cut>
  NAME=-mbs$1 MEG_RC="$RC_ALL \"model.pipeline_model_parallel_layout='$2'\" $FAST model.batch_p2p_sync=false train.micro_batch_size=$1" \
    bash bench/qwen35_27b/pp8.sh bridge
}
bridge27 1 'Et|t*9|t*9|t*9|t*9|t*9|t*10|t*8L'
bridge27 2 'Et|t*9|t*9|t*9|t*9|t*10|t*10|t*7L'
bridge27 4 'Et*2|t*9|t*10|t*9|t*10|t*9|t*10|t*5L'
bash bench/torchtitan/run.sh run qwen35_27b_cord_fsdp8_rc_bf16   # TorchTitan FSDP over 8 GPUs, every layer recomputed
bash bench/torchtitan/run.sh run qwen35_27b_cord_tp4dp2          # TorchTitan TP4 × FSDP2
```

If a rerun measures another cut (the second `stages (blocks):` line of the rdsp log), give Bridge that one. At 2 rows per microbatch, Bridge's lighter recompute settings (`recompute_method=block` with 8 or 9 layers, `recompute_granularity=selective`) ran out of memory.

| Rows per microbatch | rdsp | Megatron-Bridge | TorchTitan FSDP |
|---|---|---|---|
| 1 | 10.66 s (1,537 tokens/s per GPU) | 15.13 s (1,083) | 13.16 s (1,245) |
| 2 | 9.28 s | 9.78 s | |
| 4 | 11.33 s | 10.47 s | |
| 8 | out of memory | out of memory | |

rdsp and Bridge ran on the same machine. On another machine of the same type, rdsp with every layer recomputed (no `RC_PER`) took 11.09 s at 1 row and 10.03 s at 2 rows. Bridge at 1 row took 14.34–15.13 s across four machines. Bridge TP2 × PP2 × DP2 took 17.47 s, TorchTitan TP4 × FSDP2 38.16 s.

### Vision encoder alone

The Qwen3.5-27B encoder on one H100, 64 CORD-v2 images, every layer recomputed. From `bench/encoder`; run the Bridge line in the NeMo container with this folder mounted at `/rdsp`, as `bench/common/runs.sh` does:

```bash
mkdir -p logs && python grids.py logs/grids.json 64
python enc_bench.py --impl bridge --modes fwdbwd --recompute --grids logs/grids.json --out logs/bridge-rc.json
python enc_bench.py --impl hf --hf-attn sdpa --modes fwdbwd --recompute --grids logs/grids.json \
  --out logs/hf-rc.json
```

| Per image, median of three runs | Megatron-Bridge | Hugging Face (rdsp) |
|---|---|---|
| Forward and backward, 1 image per call | 173 ms | 114 ms |
| GPU work in that | 74 ms | 62 ms |
| The same at 8 images per call | 75 ms | 65 ms |

Without `enc_bench.py --modes fwdbwd --recompute` it also times the forward pass: 1,238 GPU kernels per image for Bridge, 782 for Hugging Face.

### MegatronMIMO

Both systems give the vision encoder one GPU and the language model the same layout, with no recompute. Qwen3.5-9B on one node, 64 rows a step:

```bash
bash bench/mimo/bridge_9b.sh prep && bash bench/mimo/mimo_9b.sh convert   # weights, rows, MIMO's checkpoint
for L in 9b_v1_tp2pp2dp1 9b_v1_tp2pp1dp2; do                              # language TP2 × PP2, then TP2 × DP2
  bash bench/mimo/mimo_9b.sh mimo $L && bash bench/mimo/mimo_9b.sh rdsp $L
done
```

These write `bench/mimo/logs/<layout>/mimo.log` and `rdsp.log`. The published TP2 × PP2 rdsp run had 26 steps (`mimo_9b.sh rdsp 9b_v1_tp2pp2dp1 26`). rdsp counts steps from 0, so its medians start at step 5 (`MEDIAN_FIRST=5`, and `MEDIAN_MIN=21` for 26 steps).

Qwen3.5-27B on 9 GPUs of two nodes, 128 rows a step, 10 steps. MIMO's example launcher runs out of memory here as shipped. `MIMO_PATCH=1` runs a patched copy (`bench/mimo/patch_mimo.py`; `bench/mimo/patch.diff` shows the edit) with BF16 gradients and the 16-bit-remainder master, as rdsp. MIMO's library is unchanged.

```bash
. ~/env2.sh
MODEL=Qwen3.5-27B RECIPE=qwen35_vl_27b_sft_16gpu_h100_bf16_config GBS=128 STEPS=50 \
  EXPORT_NAME=cord_steps-Qwen3.5-27B-128 bash bench/common/runs.sh export
MODEL=Qwen3.5-27B CONVERT_TP=4 bash bench/common/runs.sh convert-mimo
rsync -a -e "ssh -i ~/key2.pem" ~/mimo/cord_steps-Qwen3.5-27B-128 ~/mimo/Qwen3.5-27B-mimo ubuntu@$PRIV2:mimo/
bash bench/cluster/both.sh env MODEL=Qwen3.5-27B GBS=128 STEPS=10 LOGS=$HOME/rdsp/bench/mimo/logs/27b_v1_tp4pp2dp1 \
  MIMO_PATCH=1 MIMO_NAME=mimo-9gpu-10 MIMO_GPUS=9 MIMO_MBS=1 MIMO_NPROC0=8 MIMO_NPROC1=1 \
  MIMO_LANGUAGE=tp=4,pp=2,dp=1,rank_offset=0 MIMO_IMAGES=tp=1,pp=1,dp=1,rank_offset=8 \
  bash bench/common/runs.sh mimo-layout
bash bench/cluster/ray_cluster.sh up $PRIV1 $PRIV2
RDSP_ARGS=--profile bash bench/mimo/rdsp_27b.sh
```

MIMO prints its step times on node 2, in `~/both-<time>.log`: take steps 4–10 with `MEDIAN_FIRST=4 MEDIAN_MIN=5 python3 bench/common/medians.py bridge <log>`. `rdsp_27b.sh` prints its own median. The published rdsp run had `--profile` on, which synchronizes the GPU per reading.

| Model | Language layout | GPUs | rdsp | MIMO | Tokens/s per GPU, rdsp / MIMO |
|---|---|---|---|---|---|
| 9B | TP2 × PP2 | 5 | 8.63 s | 9.32 s | 3,039 / 2,814 |
| 9B | TP2 × DP2 | 5 | 7.73 s | 8.41 s | 3,391 / 3,116 |
| 27B | TP4 × PP2 | 9 | 31.23 s | 30.82 s | 933 / 945 |

## Reproducing the Qwen3.5-2B and 4B results

Results: [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md). One node with 8 GPUs of at least 32 GB (80 GB for the 4B `noncoloc` and `coloc` layouts) and about 300 GB of disk, after `bash bench/setup_node.sh`:

```bash
MODEL=Qwen3.5-2B bash bench/qwen35_small/bench.sh noncoloc tp2
MODEL=Qwen3.5-4B bash bench/qwen35_small/bench.sh tp2 tp4
```

`noncoloc` pairs MegatronMIMO with the `vision1-tp2dp2` recipe, `tp2` and `tp4` pair Megatron-Bridge with the `tp2pp2dp2` and `tp4pp2dp1` recipes, and `coloc` runs the colocated recipe alone. Logs and `summary.txt` go to `~/bench/<model>/r<round>/`.
