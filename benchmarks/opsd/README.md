# OPSD HybridEngine rollout benchmark

`benchmark_hybrid_engine_rollout.py` measures rollout-level performance for an
OPSD workload backed by DeepSpeed HybridEngine. It runs a matrix of synthetic,
exact-length prompts and reports prompt expansion, generation, prefill/decode
forward timings, generation overhead, post-processing, total latency,
generated-token throughput, and peak accelerator memory. Each case includes
raw iteration profiles, mean/p50/p95 summaries for latency metrics, and a
consistent `num_decode_forwards` count.

This benchmark depends on the rollout profiling API introduced by
DeepSpeed PR #8295:

https://github.com/deepspeedai/DeepSpeed/pull/8295

Use a DeepSpeed checkout that contains that API and place it first on
`PYTHONPATH`. The current validation scope is one process, one GPU, and ZeRO-0
with `inference_tp_size=1`.
This is a HybridEngine rollout benchmark, not a complete OPSD training-step
benchmark; it does not measure teacher inference, loss computation, backward,
or optimizer work.

## Usage

The workload matrix is controlled by `--batch-sizes`,
`--samples-per-prompt`, `--prompt-lengths`, and `--response-lengths`. Both
`--dtype fp16` and `--dtype bf16` are supported. `--warmup` and `--iterations`
control unreported warmup calls and recorded calls. Pass
`--release-inference-cache` to release the inference cache after generation;
otherwise the benchmark retains it. `--temperature`, `--top-p`, `--seed`, and
`--output` control sampling, reproducibility, and the JSON output path.
Pass `--use-shared-prefill` to enable shared prompt prefill. Shared-prefill
currently supports only ZeRO-0, `inference_tp_size=1`, and the internal KV
cache; it cannot be combined with CUDA Graph capture or
`--release-inference-cache`. The selected mode is recorded as
`use_shared_prefill` in the top-level JSON result.
Pass `--use-graph-capture` to benchmark CUDA Graph decode. CUDA Graph capture
supports only greedy generation (`--temperature 0`) and cannot currently be
combined with `--use-shared-prefill`. The first generation performs graph
capture, so use at least one warmup call before recording measurements.

The largest effective batch (`batch_size * samples_per_prompt`) executes first
so HybridEngine initializes a sufficiently large inference workspace. Results
in the output JSON retain the matrix order requested on the command line.

## Profile fields

The raw profile for every iteration contains these latency fields in
milliseconds:

- `prompt_expansion_ms`: time to expand prompts for `n_samples_per_prompt`.
- `generation_ms`: time spent in generation after prompt expansion.
- `prefill_forward_ms`: the first top-level model forward (the prefill).
- `decode_forward_ms`: cumulative time of subsequent top-level model forwards.
- `generation_overhead_ms`: `generation_ms` minus prefill and decode forward
  time, covering the remaining generation work.
- `post_processing_ms`: time to construct the rollout result after generation.
- `total_ms`: end-to-end time for the profiled rollout.

`num_decode_forwards` is metadata counting the forwards included in
`decode_forward_ms`; it is retained in each raw profile and summarized once per
case after verifying that every iteration has the same count. It is not a
latency statistic. Forward breakdown fields can be `null` on paths such as
CUDA Graph replay. If all iterations are unavailable, their case summary is
`null`; a mixture of `null` and numeric values is rejected.

For A/B comparisons, keep model, dtype, seed, workload matrix, warmup, and
iterations exactly identical between runs; change only the feature under test.

From the DeepSpeedExamples repository root, run the baseline single-GPU OPT-6.7B
FP16 benchmark with:

```bash
PYTHONPATH=/workspace/DeepSpeed_woo:/workspace/DeepSpeedExamples_woo \
torchrun --nproc_per_node=1 \
  benchmarks/opsd/benchmark_hybrid_engine_rollout.py \
  --model facebook/opt-6.7b \
  --dtype fp16 \
  --batch-sizes 1 \
  --samples-per-prompt 1 4 \
  --prompt-lengths 128 512 \
  --response-lengths 32 128 \
  --temperature 0 \
  --warmup 5 \
  --iterations 20 \
  --output /workspace/results/opsd_prefill_decode_baseline.json
```

Run the shared-prefill variant with the same workload and controls:

```bash
PYTHONPATH=/workspace/DeepSpeed_woo:/workspace/DeepSpeedExamples_woo \
torchrun --nproc_per_node=1 \
  benchmarks/opsd/benchmark_hybrid_engine_rollout.py \
  --model facebook/opt-6.7b \
  --dtype fp16 \
  --batch-sizes 1 \
  --samples-per-prompt 1 4 \
  --prompt-lengths 128 512 \
  --response-lengths 32 128 \
  --temperature 0 \
  --warmup 5 \
  --iterations 20 \
  --use-shared-prefill \
  --output /workspace/results/opsd_prefill_decode_shared.json
```

For a quick local smoke test, reduce warmup and iterations:

```bash
PYTHONPATH=/workspace/DeepSpeed_woo:/workspace/DeepSpeedExamples_woo \
torchrun --nproc_per_node=1 \
  benchmarks/opsd/benchmark_hybrid_engine_rollout.py \
  --model facebook/opt-6.7b \
  --dtype fp16 \
  --batch-sizes 1 \
  --samples-per-prompt 1 4 \
  --prompt-lengths 128 512 \
  --response-lengths 32 128 \
  --warmup 1 \
  --iterations 2 \
  --output /tmp/opsd_rollout_profile_examples.json
```

Use `--help` for the complete argument list. Model download, GPU memory, and
the fused inference kernels supported by the selected model can limit which
matrix shapes run successfully.
