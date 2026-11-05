# OPSD HybridEngine rollout benchmark

`benchmark_hybrid_engine_rollout.py` measures rollout-level performance for an
OPSD workload backed by DeepSpeed HybridEngine. It runs a matrix of synthetic,
exact-length prompts and reports prompt expansion, generation, post-processing,
total latency, generated-token throughput, and peak accelerator memory. Each
case includes raw iteration profiles, mean and p50 summaries, and p95 summaries
for latency metrics.

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

## Adaptive prefill calibration

`rollout_prefill.py` calibrates the opt-in adaptive prefill policy and compares
it with ordinary generation, equal-width CB and aligned CB. It requires a
DeepSpeed checkout containing [DeepSpeed PR #8786](https://github.com/deepspeedai/DeepSpeed/pull/8786),
installed in the active Python environment, plus PyTorch, Transformers >=4.56 and NumPy.
The scope is one GPU, greedy decoding and SDPA, with an unsharded HuggingFace
model behind the rollout interface; it does not measure a full training step.

From the DeepSpeedExamples root:

```sh
python benchmarks/opsd/rollout_prefill.py --model Qwen/Qwen3-32B --dtype bfloat16 \
  --repeats 3 --calibrate costs.json
python benchmarks/opsd/rollout_prefill.py --model Qwen/Qwen3-32B --dtype bfloat16 \
  --cost-config costs.json --repeats 3 --output sweep.json
```

Use `--model tiny-qwen2 --dtype float32` for a seeded smoke model. Calibration
uses six uniform shapes, distinct from the four mixed-length evaluation cases,
to fit non-negative fixed/token/attention costs and estimate KV and decode costs.
The built-in calibration grid reaches 768 input tokens. Extend the `shapes`
list in `calibrate()` to cover the deployment's actual lengths and feasible batch
sizes before using the estimates for longer contexts. Coefficients are specific
to the model, GPU, precision and attention backend; the sweep rejects mismatched
configuration metadata, including a source hash covering rollout, bucket planning
and static-cache implementations. Recalibrate when those implementations change.
`--cost-config` requires a complete calibration metadata object. Applications
may still pass hand-tuned coefficients directly to `HybridEngineRolloutConfig`.

Calibration runs during deployment preparation; load its saved coefficients
once when constructing the rollout. It does not profile or refit on each request:

```python
import json
from pathlib import Path
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRolloutConfig

costs = json.loads(Path("costs.json").read_text())["costs"]
cfg = HybridEngineRolloutConfig(adaptive_prefill=True, **costs)
```

The sweep reports raw timings, selected routes, output-ID agreement,
`auto_vs_best` and the worst regression. Its default two-token, EOS-disabled
generation isolates routing overhead; it is not a production rollout trace.
Use `--new-tokens` for other generation budgets and validate held-out workloads
representative of the deployment. Timing excludes loading and tokenization.

Run the fitting, MHA calibration and saved-configuration regression tests:

```sh
python -m unittest discover -s benchmarks/opsd -p test_rollout_prefill.py
```
