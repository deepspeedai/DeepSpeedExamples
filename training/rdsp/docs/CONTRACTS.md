# Contracts

What rdsp's public API guarantees, what it raises, and what it does not support. The docstrings in `src/ray_deepspeed_pipeline/` carry the same contracts at the call site.

## Public API

```python
import ray_deepspeed_pipeline as rdsp
```

| Name | Guarantees | Raises |
|---|---|---|
| `rdsp.initialize(model=, config=, loss_fn=, pipeline_config=, weights=None, training_data=None)` | Returns `(engine, None, dataloader_or_None, None)`. Every check, including the cluster's GPU count, runs before any Ray actor starts; only placement is checked when the actors start: each stage on one node, or, for a stage wider than a node, every tensor-parallel group on one node. With `weights`, each stage loads only its own tensors from the HF checkpoint. | `ValidationError`, `UnsupportedInV1` |
| `rdsp.PipelineConfig` | Lowered once into an immutable plan. The plan's hash binds checkpoints. | `ValidationError` |
| `rdsp.StageOverride(stage, num_gpus, tp, sp, ep, zero_stage, recompute, recompute_blocks, offload_optimizer, compile, compile_vision)` | One stage's grid: `num_gpus = dp × sp × tp`, with dp derived. Stages without an override get 1 GPU. | `ValidationError` (grid does not divide, ep does not divide `num_gpus`, tp does not divide the key/value heads) |
| `rdsp.ConnectionOverride(source, dest, conversion)` | The conversion on an edge is derived from the two layouts; a declared one must match. | `ValidationError` |
| `rdsp.UniformTransformerBlocks()` | Blocks split as evenly as the counts allow. | `ValidationError` |
| `rdsp.BalancedTransformerBlocks(vision_token_ratio=, vision_stage=False, measured=None)` | Cut that minimises the most expensive stage, from parameter counts scaled by tokens processed. Measure the ratio with `partition.vision_token_ratio(config, batch)`. `vision_stage=True` puts the vision encoder alone on stage 0. `measured=` re-cuts by what `engine.measure_stages()` measured: stage times, and memory as a limit. | `ValidationError` |
| `rdsp.MeasuredStages` | What `engine.measure_stages()` returns: per stage, the cut it ran with, its busy time per step, its peak memory and its bytes of state per parameter. | `ValidationError` |
| `rdsp.ExplicitCuts((c1, ...))` | Stage i+1 starts at block `c_i`. A first cut of 0 gives the vision encoder its own stage. | `ValidationError` |
| `rdsp.ColocatedVision(recompute, compile, encode_per_microbatch)` | The encoder runs on every GPU. fp32 master weights and optimizer state are split over all ranks, and checkpoints hold the full state. | `ValidationError` (encoder output enters the decoder blocks) |
| `rdsp.TokenMeanLoss(sum_fn, count_fn, per_microbatch=False)` | The step loss is the mean over every counted token of the step. With `per_microbatch=True`, training averages per rank-microbatch (Megatron's default) and the report is still the token mean. | — |
| `rdsp.next_token_loss_sum(logits, labels)` | Summed next-token cross entropy, on full logits or on a vocab-parallel head's shard. As the loss, it lets TP stages skip gathering the 248k-wide logits. | — |
| `engine.train_batch(data_iter)` | Consumes exactly M entries (M = microbatches per step). Returns the step loss. After `StepFailed`, the next call with the same iterator (or the engine's loader) reruns the failed step's entries instead of reading new ones. | `StepFailed`, `PipelinePoisoned` |
| `engine.eval_batch(data_iter)` | Forward only. Weights unchanged. Takes its own `data_iter` even when the engine owns a loader; reads the loader only when `data_iter` is omitted. | `StepFailed`, `PipelinePoisoned` |
| `engine.save_checkpoint(dir, tag)` | All stages, then the manifest, written last and atomically. A failed save commits nothing. A stage that spans nodes needs `dir` on storage its nodes share; otherwise the save is refused. The engine-owned loader's saved position counts only entries trained on. | `CheckpointError` |
| `engine.load_checkpoint(dir, tag)` | Verifies the manifest and every file's SHA-256, rebuilds dead actors, then loads. Clears a poisoned pipeline. | `CheckpointError` |
| `engine.stage_blocks` | Each stage's `[start, stop)` transformer blocks. | — |
| `engine.measure_stages(data_iter, steps=3, skip=1)` | Runs `steps` training steps with every stage timed and returns an `rdsp.MeasuredStages` from the steps after the first `skip`. The steps update the weights like any step. To use the result, shut this engine down and build a new one with `BalancedTransformerBlocks(measured=...)`. | `ValidationError`, `StepFailed`, `PipelinePoisoned` |
| `engine.shutdown()` | Stops every stage's workers and releases their GPUs. The engine is unusable afterwards. | — |

## Data

- An entry is `(inputs, labels)`. Inputs are a tensor or a dict of tensors; vision models pass `pixel_values` and `image_grid_thw` as one tensor per row.
- A step reads exactly M entries and leaves the rest of the iterator untouched.
- The engine-owned loader (`initialize(training_data=...)`) drops a short last entry, so every microbatch has the same rows.
- Microbatches in one step may differ in sequence length. The stages then send each boundary with its shape.
- With `prefetch=True`, the next step's M entries are read while the current step runs. Pass the same iterator to every `train_batch` call. `initialize(training_data=...)` rejects prefetch. `load_checkpoint()` drops the step read ahead.
- Under prefetch, `save_checkpoint()` runs with the next step's M entries already read: a position you track yourself is M entries past the saved step, so resume from M entries earlier. The manifest records the count as `data_position["read_ahead"]`.

## Loss

- `loss_fn(outputs, labels)` returns a scalar for one rank's rows of one microbatch. The step loss is the mean over microbatches and data-parallel ranks.
- A loss with `takes_vocab_shards = True` (`next_token_loss_sum`, or a `TokenMeanLoss` built on it) may receive a TP rank's slice of the vocabulary. The logits tensor then carries `.vocab_shard`.

## Failures

| When | Raises | State after |
|---|---|---|
| A stage fails before any stage applies its update | `StepFailed` | Weights unchanged; links rebuilt. A retry with the same iterator (or the engine's loader) reruns the failed step's entries, discards leftovers and rewinds the RNG, so it is exact. A different iterator starts a new step. |
| A stage fails while updates are being applied | `PipelinePoisoned` | Every call is refused until `load_checkpoint()`. |
| A checkpoint fails verification | `CheckpointError` | Nothing loaded. |

A checkpoint loads only into a pipeline with the same plan hash: same model split, layouts, microbatches and DeepSpeed config (including optimizer settings such as the learning rate). Settings that change how a step runs but not what a stage saves are left out of the hash: `prefetch`, `recompute`, `recompute_blocks`, `compile`, `compile_vision`, `encode_per_microbatch`.

## DeepSpeed settings rdsp changes

Every stage runs a stock DeepSpeed engine, with these settings changed:

- `gradient_accumulation_steps = 1`: the coordinator accumulates over the step's microbatches.
- `gradient_clipping = 0`: DeepSpeed would clip each stage by its own norm, not the global one. A nonzero value is rejected.
- `train_batch_size` is dropped: each stage's DeepSpeed derives it from its own grid.
- `bf16.immediate_grad_update = true` under bf16 with fp32 accumulation and ZeRO-1, unless the config sets it. Each bf16 gradient is added into fp32 as soon as it exists and then freed: 2 fewer bytes of state per parameter. rdsp does the add itself, in one operation, with the same result as DeepSpeed's.
- `rdsp_master_weights: "remainder"` (rdsp's own key, taken out before DeepSpeed reads the config): the stage's optimizer becomes rdsp's AdamW, whose exact fp32 master weights are stored as the bf16 weights plus a 16-bit remainder. It needs bf16, ZeRO 1 or 2 without offload, bf16 gradient accumulation and an AdamW optimizer section; anything else raises `ValidationError`.
- `rdsp_optimizer_group_numel: N` (rdsp's own key): the stage's parameters reach the optimizer in groups of at most N elements, which lowers the peak memory of ZeRO's step. Stages with expert parallelism ignore it: DeepSpeed builds their groups.

## Supported layouts

Validated against an unsplit model (loss parity, gradient parity, checkpoint round trip, rank-kill recovery). Evidence is in `src/ray_deepspeed_pipeline/support_matrix.py`.

| Row | Stages |
|---|---|
| two-stage-baseline | 1 GPU, 1 GPU (Qwen3-0.6B, bf16) |
| static-boundary | 1 GPU, dp=2, 1 GPU |
| resource-mesh | dp=2, dp=4, dp=2 |
| dp-zero | dp=4 ZeRO-2, dp=2 ZeRO-1, dp=2 ZeRO-0 |
| autotp | 1 GPU, tp=2, tp=2×dp=2 |
| autoep-folding | 1 GPU, ep=4, ep=2 folded with tp=2 |
| sequence-parallel | sp=2, sp=2×dp=2, 1 GPU |

The vision-language layouts benchmarked against Megatron (Qwen3.5-2B/4B) are listed in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).

## Not supported

- TP above, or not dividing, the model's key/value heads (Qwen3.5-2B has 2: no TP4). Raises `ValidationError`.
- Parameters tied across stages. Untie them, or keep both users on one stage.
- Colocated vision for encoders whose features enter the decoder blocks (Qwen3-VL deepstack). The check reads `deepstack_visual_indexes`, so other ways of injecting features are not detected.
- fp16. Each stage's DeepSpeed would scale its backward by its own loss scale, so upstream gradients would carry every downstream scale. Raises `ValidationError`; use bf16 or fp32.
- Sequence parallelism (`sp > 1`) for models with linear-attention layers (Qwen3.5): Ulysses splits only the attention DeepSpeed swaps in. Raises `ValidationError`.
- Sequence parallelism on a stage whose blocks get a 4D attention mask (eager attention, or a padding mask): the mask cannot be split by sequence. The step raises `StepFailed`.
- The 4-node composed layout (`four-stage-mixed`) is validated on one node only.
