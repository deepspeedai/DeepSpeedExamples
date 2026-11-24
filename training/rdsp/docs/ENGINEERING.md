# How rdsp works

rdsp splits a model's stack of blocks into contiguous stages, places each stage on its own group of GPUs, and trains the result so it computes what the unsplit model would. Each GPU runs a Ray actor holding a stock DeepSpeed engine for its stage's slice. Activations go forward and gradients go back over NCCL.

Why split by depth: only activations cross a cut, once per microbatch, so it is the cheapest split across slow links. And stages do unequal work (a vision encoder, an output head over a 248k vocabulary), so each stage gets its own GPU count and layout.

## Design rules

**Stock DeepSpeed and Ray.** DeepSpeed is pinned to one commit and installed unmodified; every stage calls `deepspeed.initialize()`. rdsp crosses into DeepSpeed internals in exactly three places, all in `deepspeed_adapter.py`:

- `_free_bf16_grads_after_accumulating` wraps one bf16 optimizer method on the instance to free each bf16 gradient once it's folded into fp32 (see Memory). It also does the fold itself, straight from bf16 into the fp32 buffer: one operation where DeepSpeed converts and then adds, with the same result.
- `_master_in_model_weights` (the remainder master) points each ZeRO partition's master weights at the model's own bf16 partition, so no separate master copy is kept.
- `reset()` clears two private ZeRO running sums that `zero_grad()` leaves behind, so a retried step doesn't add onto an abandoned one.

All three depend on the pinned commit (unit tests in `tests/unit/test_deepspeed_adapter_grads.py` and `tests/unit/test_remainder_master_engine.py`, the abandoned-step GPU test); a DeepSpeed upgrade has to re-run the GPU suite.

**The driver never touches tensors.** The driver, meaning the user's process, holds no weights and never receives an activation or gradient. Autograd graphs can't leave their process, so whichever GPU ran a forward runs its backward, the loss is computed on the last stage (labels travel there), and only losses and statuses return to the driver.

**The API doesn't pretend.**
- `rdsp.initialize()` mirrors `deepspeed.initialize()`, but returns `None` for the optimizer: the real optimizers live in the actors, and a look-alike object whose changes do nothing would be worse.
- `engine.forward/backward/step` raise and point at `train_batch()`: one step interleaves hundreds of operations across stages, so there's no single "the forward" for them to mean.

## Planning

`rdsp.initialize()` lowers `PipelineConfig` into an immutable plan (`compiler.lower`):

- the cuts;
- each stage's grid, `gpus = dp × sp × tp`;
- each stage's DeepSpeed config;
- the conversion on every edge between stages;
- a hash that binds checkpoints.

Every check runs here, before any actor starts (the cluster's GPU count just after, still before any actor). TP must divide the key/value heads, rows must divide every stage's DP degree, parameters can't be tied across stages, and so on. Settings that change how a step runs but not what a stage saves (`prefetch`, `recompute`, `compile`, `compile_vision`, `encode_per_microbatch`) are left out of the hash, so toggling them keeps checkpoints loadable.

The splitter finds the model's longest `ModuleList` of same-class blocks; a vision encoder's blocks are never cut. Everything before the blocks goes to the first stage, everything after them to the last.

With `weights=<HF checkpoint>`, the driver model is built empty and each stage reads only its own tensors from the safetensors files.

## One training step

1. The driver takes exactly M `(inputs, labels)` entries. Too few raises `StepFailed` with nothing dispatched; extra entries stay in the iterator.
2. Each GPU gets one Ray call: its stage's whole 1F1B op list for the step. Stage *s* of *N* runs `min(N−1−s, M)` warm-up forwards, then alternates.
3. Forward: receive from the overlapping upstream ranks, assemble, `x.detach().requires_grad_()`, run the engine, send downstream asynchronously.
4. Backward: receive the gradient `g` for this stage's output, then call `engine.backward((out * g).sum())`. Differentiating that scalar reproduces `g` exactly, and it is the only form DeepSpeed's backward accepts. Send `x.grad` upstream. The last stage backpropagates its loss.
5. Only the M-th backward is a DeepSpeed accumulation boundary. Intra-stage gradient reduction happens once per step.
6. Each rank reports *ready*. Only when every rank of every stage is ready does the driver send *apply*, and each engine steps exactly once.

Two NCCL groups span all ranks of all stages: `fwd` for activations and `bwd` for gradients. They sit next to each stage's own DeepSpeed world.

An earlier design dispatched every forward and backward from the driver as a separate Ray call. At about 5 ms of Ray overhead per call, that overhead set the pace. The current path is checked against losses recorded from the old one (`test_losses_match_recorded_reference`).

**Failures.**
- A failure before any apply raises `StepFailed`. Weights are unchanged, links are rebuilt, and on retry each rank discards leftovers (`reset()`) and rewinds its RNG, so the retry equals a clean step.
- A failure during apply means some stages updated and others didn't. The engine raises `PipelinePoisoned` and refuses every call until `load_checkpoint()`.

## Stages with different layouts

A stage can use several GPUs in four ways, each an ordinary DeepSpeed feature running inside that stage alone:

| | Each GPU gets |
|---|---|
| DP (any ZeRO stage) | different rows |
| TP (AutoTP) | the same rows, a slice of every weight matrix |
| SP (Ulysses) | the same rows, a different stretch of the sequence (not for models with linear-attention layers, such as Qwen3.5) |
| EP (AutoEP), optionally folded onto TP | different rows, some of the experts |

**Boundaries.** A rank's cell is its block of rows (`rows / dp`) and of sequence (`seq / sp`); TP ranks share a cell. Each receiving rank asks for exactly the upstream pieces that overlap its cell and stitches them together. Only TP rank 0 of a cell sends, and nothing is gathered in one place. Gradients take the same routes in reverse, scaled by `dp_s / dp_(s+1)`: DeepSpeed averages over DP and sums over SP, so without the factor a stage learns at the wrong rate.

**Shapes.** Each message carries a small header with dtype and shape. Normally a header is sent once per step per peer. When a step's microbatches differ in length (per-microbatch padding), every message carries its own.

## Running Hugging Face models without per-model code

`HFModelStage` runs the model's own `forward` on every stage:

- Blocks outside the stage return their input.
- Modules whose weights the stage doesn't own return their input, or zeros for an embedding.
- A pre-hook on each block injects what arrived from upstream.
- The stage's output is whatever the loop hands the next block. Work between blocks counts; work after the last block (norm, head) belongs to the last stage.

The boundary carries the hidden state plus the tensors the model passes to each block: rotary tables, position ids, masks. For Qwen3-VL-32B that's about 5% over the hidden state.

TP uses the model's AutoTP plan. For configs without one, rdsp generates a plan for the standard `q/k/v/o_proj` and `gate/up/down_proj` names, the vision blocks' `attn.qkv`/`attn.proj`/`mlp.linear_fc1/fc2`, and Qwen3.5's linear-attention projections.

Two pieces sit outside AutoTP and split the way Megatron does:
- **Vocabulary** (`vocab_parallel.py`): each TP rank keeps 1/tp of the embedding's and the output head's rows.
- **Qwen3.5 linear attention** (`linear_attention_parallel.py`): each TP rank keeps 1/tp of the key heads and of the value heads that read them, with their projections, conv filter, `A_log` and `dt_bias`. The heads' recurrences are independent, so the only communication is a sum of `out_proj`'s output, a sum of the input gradient, and a sum of the shared gated norm's weight gradient. AutoTP's own rule for these projections (`colwise_gather_output`) would split the projections but leave every rank holding every head's conv filter and running every head's recurrence. A layer whose heads don't divide by TP stays whole.

## Vision encoder placement

| Placement | How | Encoder runs on |
|---|---|---|
| First stage | default | the first stage's GPUs, in its layout |
| Vision-only stage | first cut at 0 (`--cuts 0`) | its own stage, with its own GPU count and layout |
| Colocated | `PipelineConfig(colocated_vision=rdsp.ColocatedVision())` | every GPU, data parallel |

**Vision-only stage.** This stage holds the encoder and nothing else. It sends image embeddings and rotary tables, and the next stage embeds the text itself. This is MegatronMIMO's layout. It freed enough memory to drop recompute and took the 2B MIMO-layout step from 13.51 s to 11.3 s.

**Colocated** (`vision.py`).
1. Each rank encodes its share of the step's images and sends each image's features to the first-stage ranks whose rows hold it.
2. The language pipeline runs its normal 1F1B.
3. The first stage returns each image's feature gradient to the rank that encoded it, and the encoder's gradients are reduced over all ranks.

The encoder trains with a torch optimizer built from the DeepSpeed config, with fp32 master weights and optimizer state split over all ranks.

- `encode_per_microbatch` encodes just ahead of the pipeline's need instead of all at the start, keeping graphs for about 2 × stages microbatches instead of the whole step.
- Colocation only works for encoders whose output enters through the input embeddings (Qwen3.5). Qwen3-VL's deepstack feeds its first blocks, so it keeps the encoder on the first stage.
- It matches the unsplit model, but on 32 GB GPUs it's slower than the first-stage placement (4B: 9.66 s vs 8.69 s).

## Balanced cuts

`BalancedTransformerBlocks` picks the contiguous split that minimises the most expensive stage, each stage's cost divided by its GPU count. Ties go to the lowest sum of squared stage costs, so no stage idles.

- A decoder block costs its parameter count.
- The output head, on the last stage, costs the same way.
- Embeddings are lookups and cost nothing.
- The vision encoder costs its parameter count times `vision_token_ratio`, measured from the first batch by `partition.vision_token_ratio`:

```
ratio = Σ_images patches × (1 + 2 × patches × width / layer_params) / text tokens computed
```

The second term in the sum is the encoder's attention over each image: about 4 × patches × width multiply-adds per patch, against 2 × layer_params for its matrix products. The denominator counts text tokens at each microbatch's padded length, not the configured maximum.

The estimate takes milliseconds and runs no trial steps. It picks the fastest measured cut in both TP2×PP2×DP2 layouts and is one layer off under TP4 ([BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)). It ignores communication and counts MoE experts in full. `ExplicitCuts` overrides it.

## Memory

- **2 + 4 + 12/dp bytes per parameter** of training state under bf16 + ZeRO-1 + fp32 accumulation: bf16 weights, fp32 accumulator, then fp32 master and Adam's two moments split over DP (12 bytes at DP=2). rdsp sets `bf16.immediate_grad_update` so each bf16 gradient is added into fp32 as soon as autograd produces it, then frees the bf16 copy; DeepSpeed alone only zeroes it, keeping 2 more bytes per parameter. Megatron spends the same. On 4B this removed the out-of-memory errors that had forced recompute.
- **Per-microbatch padding.** With `--pad-per-microbatch`, each microbatch is padded to its own longest row, like Megatron's collator, instead of the whole step being padded to its longest row. On CORD-v2 this cut padded tokens from 1.71 to 1.20 per real token.
- **Recompute** (`recompute=1` per stage, or `ColocatedVision(recompute=True)`) keeps only each block's input and recomputes the rest in backward.

## Data prefetch

With `PipelineConfig(prefetch=True)`, the driver reads the next step's M entries while the current step runs and ships them to the stages ahead of time. For 2B on CORD-v2 that's about 576 MB of images per step, and handing a step to the GPUs fell from 1.1 s to 1 ms.

- The caller passes the same iterator to every `train_batch()`.
- `initialize(training_data=...)` rejects prefetch, since rdsp would own the iterator.
- `load_checkpoint()` drops the step that was read ahead.

## Loss

- **Default.** `loss_fn(outputs, labels)` returns one rank's loss for one microbatch, and the step loss is the mean over microbatches and DP ranks.
- **Token mean.** `TokenMeanLoss(sum_fn, count_fn)` averages over every counted token of the step instead, like Megatron's `calculate_per_token_loss`. The driver counts the step's tokens from the labels before dispatch, so each backward already carries its final weight. `per_microbatch=True` trains on per-microbatch means (MegatronMIMO's only mode) and still reports the token mean.
- **Split-vocabulary loss.** `next_token_loss_sum` declares `takes_vocab_shards`. On a TP last stage, each rank then keeps its slice of the logits, and the ranks exchange the max, the sum of exponentials and the target logit per token, instead of gathering the full 248k-wide logits (2 GB per microbatch for Qwen3.5). Megatron does the same.

## Checkpoints

Each stage saves with DeepSpeed's own `save_checkpoint`, plus each rank's RNG state. The driver then writes `manifest.json` last and atomically. It holds the plan hash, the step, the data position, and the SHA-256 of every file. A directory without a manifest is an unfinished save and is never loaded.

`load_checkpoint()` works in four steps:
1. Verify the manifest and plan hash.
2. Rebuild every actor if any died, preferring the same nodes so local files are still there.
3. Have each stage verify its own files on its node.
4. Only when every stage passes, load.

A test trains, saves and keeps training, then loads into a pipeline built from different random weights; its losses match bit for bit.

## How we know it works

One yardstick at every level: the split model must produce the unsplit model's numbers.

- **CPU suite** (~430 tests). The full runtime runs on Ray with a stub engine:
  - schedules and boundary routing;
  - exact data consumption;
  - the failure contract;
  - checkpoint round trips;
  - 24 HF families trained through `rdsp.initialize()` against the unsplit model;
  - the import-time guarantee that `import ray_deepspeed_pipeline` pulls in neither Ray nor DeepSpeed.
- **GPU suite on Modal.** Every supported layout in [CONTRACTS.md](CONTRACTS.md) with real DeepSpeed and NCCL, on fp32 tiny models. Each run checks:
  - loss parity within 1e-4;
  - per-parameter update parity;
  - checkpoint resume;
  - a killed rank recovering through `load_checkpoint()`.

**Why SGD, not Adam, in parity tests.** The first parity test used Adam, which normalizes away a uniformly wrong gradient scale: a stage receiving half its true gradient still trained identically, and a deliberately broken build passed. The tests now use SGD with momentum, and refuse to run unless each step moves the loss by at least 20× the tolerance. That stricter test found four real bugs, all fixed:

- Under ZeRO-1/2, DeepSpeed replaced instead of accumulating each microbatch's sharded gradient.
- Qwen3's per-head q/k norms drifted across TP ranks.
- The SP gradient scale was applied inverted.
- AutoEP folding sent upstream one rank's partial gradient instead of the average.

## Honest edges

- **SP.**
  - Never on the last stage: its labels would straddle ranks.
  - Never on the first stage of a vision model: image positions need the whole sequence.
- **TP and SP** can't share a stage (a DeepSpeed limitation).
- **Placement.** One stage fits on one node, unless it needs more GPUs than a node has: then it spans nodes with each tensor-parallel group inside one node, and its checkpoints need storage those nodes share.
- **Checkpoints.** Node-local: they survive actor loss, not node loss.
- **No global gradient clipping.** DeepSpeed would clip each stage by its own norm, a different algorithm, so a nonzero `gradient_clipping` is rejected.
- **Untested at scale.**
  - The multi-node path is tested on Ray, not on real multi-node hardware.
  - The 32-GPU four-stage layout hasn't run: the Modal account caps at 10 GPUs.

## Profiling

- `RDSP_PROFILE=1` (or `train_vl.py --profile`) prints, per rank and step, the time in forward, backward, waiting on neighbours and the optimizer step. It synchronizes the GPU per reading, so steps run slower.
- `RDSP_TORCH_PROFILE=<step>` (0-based) writes a torch profiler kernel table for that step to `/tmp/rdsp_torch_profile_stage<s>_rank<r>.txt`.

## Running the tests

```bash
python demo.py           # CPU, ~30 s: pipeline vs unsplit losses side by side
pytest -q                # CPU suite; GPU tests skip
modal run scripts/modal_tests.py --gpus L4:2 --tests tests/integration/test_two_stage_baseline.py
modal run scripts/modal_tests.py --gpus L4:2 --tests tests/integration/test_checkpoint_gpu.py
modal run scripts/modal_tests.py --gpus L4:8 --tests tests/integration/test_heterogeneous_pipeline.py
```

GPU runs are billed.
