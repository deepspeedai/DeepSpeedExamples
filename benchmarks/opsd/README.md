# OPSD HybridEngine rollout benchmark

`benchmark_hybrid_engine_rollout.py` measures rollout-level performance for an
OPSD workload backed by DeepSpeed HybridEngine. It runs a matrix of synthetic,
exact-length prompts and reports prompt expansion, generation, post-processing,
total latency, generated-token throughput, and peak accelerator memory. Each
case includes raw iteration profiles plus mean, p50, and p95 summaries.

This benchmark depends on the rollout profiling API introduced by
DeepSpeed PR #8295:

https://github.com/deepspeedai/DeepSpeed/pull/8295

Use a DeepSpeed checkout that contains that API and place it first on
`PYTHONPATH`. The current validation scope is one process, one GPU, and ZeRO-0.
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

The largest effective batch (`batch_size * samples_per_prompt`) executes first
so HybridEngine initializes a sufficiently large inference workspace. Results
in the output JSON retain the matrix order requested on the command line.

From the DeepSpeedExamples repository root, run a single-GPU benchmark with:

```bash
PYTHONPATH=/workspace/DeepSpeed_woo:/workspace/DeepSpeedExamples \
torchrun --nproc_per_node=1 \
  benchmarks/opsd/benchmark_hybrid_engine_rollout.py \
  --model facebook/opt-6.7b \
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
