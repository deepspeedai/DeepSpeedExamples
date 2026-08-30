# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Reproducible OPSD continuous-batching benchmark.

The benchmark compares independent Hugging Face generation, capacity-bounded
static batches, and :meth:`HybridEngineRollout.generate_continuous`.  Prompts
are deterministic token IDs; tokenization and model loading are outside the
timed region.
"""

import argparse
import json
import math
import statistics
import time
from pathlib import Path


def percentile(values, quantile):
    """Return the OPSD percentile (nearest-rank, ceil-based) used elsewhere."""
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return ordered[index]


def summarize_latencies(latencies):
    """Summarize measured milliseconds with the existing OPSD percentile rule."""
    if not latencies:
        raise ValueError("latencies must not be empty")
    return {
        "mean": statistics.mean(latencies),
        "p50": percentile(latencies, 0.50),
        "p95": percentile(latencies, 0.95),
    }


def group_request_indices(response_lengths, max_batch_size):
    """Group requests in input order into capacity-sized static batches."""
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    return [tuple(range(start, min(start + max_batch_size, len(response_lengths))))
            for start in range(0, len(response_lengths), max_batch_size)]


def static_computed_tokens(response_lengths, max_batch_size):
    """Count tokens computed by static batching, including short-request extras."""
    return sum(len(group) * max(response_lengths[index] for index in group)
               for group in group_request_indices(response_lengths, max_batch_size))


def validate_args(args):
    if args.dtype not in ("fp16", "bf16"):
        raise ValueError("dtype must be fp16 or bf16")
    if args.prompt_length <= 0 or args.max_batch_size <= 0:
        raise ValueError("prompt-length and max-batch-size must be positive")
    if not args.response_lengths or any(length <= 0 for length in args.response_lengths):
        raise ValueError("response-lengths must contain positive lengths")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if args.temperature != 0.0:
        raise ValueError("this benchmark currently supports greedy generation only (temperature=0)")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")


def _build_parser():
    parser = argparse.ArgumentParser(description="Benchmark OPSD continuous batching")
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--response-lengths", type=int, nargs="+", default=[32, 64, 96, 128])
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="opsd_continuous_batching.json")
    return parser


def _synthetic_prompts(vocab_size, pad_token_id, prompt_length, count, device, seed=0):
    """Build deterministic, equal-width prompts without pad IDs."""
    if vocab_size <= 1:
        raise ValueError("model vocabulary must contain at least two tokens")
    available = [token for token in range(vocab_size) if token != pad_token_id]
    if not available:
        raise ValueError("model vocabulary has no token distinct from pad_token_id")
    values = [available[(seed + offset * prompt_length + position) % len(available)]
              for offset in range(count) for position in range(prompt_length)]
    import torch
    return [torch.tensor(values[offset * prompt_length:(offset + 1) * prompt_length],
                         dtype=torch.long, device=device).unsqueeze(0)
            for offset in range(count)]


def _sampling_config(length, temperature):
    from deepspeed.runtime.rollout.base import SamplingConfig
    return SamplingConfig(max_new_tokens=length, temperature=temperature, top_p=1.0, n_samples_per_prompt=1)


def _request(prompt_ids):
    from deepspeed.runtime.rollout.base import RolloutRequest
    import torch
    return RolloutRequest(prompt_ids=prompt_ids, prompt_attention_mask=torch.ones_like(prompt_ids))


def _sync():
    import torch
    torch.cuda.synchronize()


def _reset_peak_memory():
    import torch
    torch.cuda.reset_peak_memory_stats()


def _peak_memory_mb():
    import torch
    return torch.cuda.max_memory_allocated() / (1024**2)


def _response(output_ids, prompt_length, response_length):
    return output_ids[0, prompt_length:prompt_length + response_length].detach().cpu()


def _assert_matches(reference, candidate, mode, request_index):
    if len(reference) != len(candidate) or not reference.equal(candidate):
        raise AssertionError(f"{mode} request {request_index} response tokens differ from sequential_eager")


def _run_sequential(generate, model_config, requests, response_lengths, args, prompt_length):
    latencies = []
    reference_responses = []
    pad_token_id = getattr(model_config, "pad_token_id", None)
    for _ in range(args.warmup):
        for request, length in zip(requests, response_lengths):
            generate(request.prompt_ids, attention_mask=request.prompt_attention_mask,
                     max_new_tokens=length, min_new_tokens=length, eos_token_id=None,
                     do_sample=False, pad_token_id=pad_token_id)
    _reset_peak_memory()
    for _ in range(args.iterations):
        _sync()
        start = time.perf_counter()
        outputs = []
        for request, length in zip(requests, response_lengths):
            output = generate(request.prompt_ids, attention_mask=request.prompt_attention_mask,
                              max_new_tokens=length, min_new_tokens=length, eos_token_id=None,
                              do_sample=False, pad_token_id=pad_token_id)
            outputs.append(_response(output, prompt_length, length))
        _sync()
        latencies.append((time.perf_counter() - start) * 1000.0)
        if not reference_responses:
            reference_responses = outputs
        else:
            for index, response in enumerate(outputs):
                _assert_matches(reference_responses[index], response, "sequential_eager", index)
    return latencies, reference_responses, _peak_memory_mb()


def _run_static(generate, model_config, requests, response_lengths, args, prompt_length, reference_responses):
    latencies = []
    groups = group_request_indices(response_lengths, args.max_batch_size)
    pad_token_id = getattr(model_config, "pad_token_id", None)

    def invoke():
        outputs = [None] * len(requests)
        for group in groups:
            max_length = max(response_lengths[index] for index in group)
            prompt_ids = __import__("torch").cat([requests[index].prompt_ids for index in group], dim=0)
            attention = __import__("torch").cat([requests[index].prompt_attention_mask for index in group], dim=0)
            generated = generate(prompt_ids, attention_mask=attention, max_new_tokens=max_length,
                                 min_new_tokens=max_length, eos_token_id=None, do_sample=False,
                                 pad_token_id=pad_token_id)
            for row, index in enumerate(group):
                outputs[index] = generated[row, prompt_length:prompt_length + response_lengths[index]].detach().cpu()
        return outputs

    for _ in range(args.warmup):
        invoke()
    _reset_peak_memory()
    for _ in range(args.iterations):
        _sync()
        start = time.perf_counter()
        outputs = invoke()
        _sync()
        latencies.append((time.perf_counter() - start) * 1000.0)
        for index, response in enumerate(outputs):
            _assert_matches(reference_responses[index], response, "static_batch", index)
    return latencies, _peak_memory_mb()


def _run_continuous(rollout, requests, response_lengths, args, prompt_length, reference_responses):
    latencies = []
    configs = [_sampling_config(length, args.temperature) for length in response_lengths]
    for _ in range(args.warmup):
        rollout.generate_continuous(requests, configs, max_batch_size=args.max_batch_size)
    _reset_peak_memory()
    for _ in range(args.iterations):
        _sync()
        start = time.perf_counter()
        outputs = rollout.generate_continuous(requests, configs, max_batch_size=args.max_batch_size)
        _sync()
        latencies.append((time.perf_counter() - start) * 1000.0)
        for index, (output, length) in enumerate(zip(outputs, response_lengths)):
            response = output.input_ids[0, prompt_length:prompt_length + length].detach().cpu()
            _assert_matches(reference_responses[index], response, "continuous_batch", index)
    return latencies, _peak_memory_mb()


def _mode_result(latencies, useful_tokens, computed_tokens, peak_memory):
    summary = summarize_latencies(latencies)
    mean_latency = summary["mean"]
    return {
        "latency_ms": summary,
        "useful_tokens": useful_tokens,
        "computed_tokens": computed_tokens,
        "useful_tokens_per_second": useful_tokens / (mean_latency / 1000.0),
        "peak_memory_mb": peak_memory,
    }


def _change_percent(value, baseline):
    return (value - baseline) / baseline * 100.0


def _build_result(args, environment, mode_results, useful_tokens):
    sequential = mode_results["sequential_eager"]
    continuous = mode_results["continuous_batch"]
    static = mode_results["static_batch"]
    return {
        "environment": environment,
        "config": {"model": args.model, "dtype": args.dtype, "prompt_length": args.prompt_length,
                    "response_lengths": args.response_lengths, "max_batch_size": args.max_batch_size,
                    "warmup": args.warmup, "iterations": args.iterations,
                    "temperature": args.temperature, "seed": args.seed},
        "results": mode_results,
        "comparisons": {
            "continuous_vs_sequential": {
                "latency_change_percent": _change_percent(continuous["latency_ms"]["mean"],
                                                             sequential["latency_ms"]["mean"]),
                "useful_throughput_change_percent": _change_percent(continuous["useful_tokens_per_second"],
                                                                     sequential["useful_tokens_per_second"]),
            },
            "continuous_vs_static": {
                "latency_change_percent": _change_percent(continuous["latency_ms"]["mean"],
                                                           static["latency_ms"]["mean"]),
                "useful_throughput_change_percent": _change_percent(continuous["useful_tokens_per_second"],
                                                                     static["useful_tokens_per_second"]),
                "avoided_decode_tokens": static["computed_tokens"] - continuous["computed_tokens"],
            },
        },
    }


def run(args):
    validate_args(args)
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import deepspeed
    from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig

    if not torch.cuda.is_available():
        raise RuntimeError("continuous batching benchmark requires CUDA")
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("this benchmark supports exactly one GPU process")
    torch.manual_seed(args.seed)
    device = torch.device("cuda", int(os.getenv("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define pad_token_id or eos_token_id")
        pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    # The Core continuous scheduler otherwise treats model EOS as early completion.
    tokenizer.eos_token_id = None
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.eval()
    model.config.pad_token_id = pad_token_id
    model.config.eos_token_id = None
    capacity = args.max_batch_size
    # Keep all three modes on the same unmodified HF forward/generate path.
    # Enabling DeepSpeed HybridEngine here replaces ``module.generate`` and
    # injects inference kernels whose batch-shape-dependent numerics can make
    # exact token comparison impossible. Core's rollout API only requires an
    # object exposing ``module`` for continuous generation.
    from types import SimpleNamespace
    engine = SimpleNamespace(module=model)
    rollout = HybridEngineRollout(engine, tokenizer, HybridEngineRolloutConfig())
    prompts = _synthetic_prompts(model.config.vocab_size, pad_token_id, args.prompt_length,
                                 len(args.response_lengths), device, args.seed)
    requests = [_request(prompt) for prompt in prompts]

    hf_generate = model.generate
    sequential_latencies, references, sequential_peak = _run_sequential(
        hf_generate, engine.module.config, requests, args.response_lengths, args, args.prompt_length)
    static_latencies, static_peak = _run_static(
        hf_generate, engine.module.config, requests, args.response_lengths, args, args.prompt_length, references)
    continuous_latencies, continuous_peak = _run_continuous(
        rollout, requests, args.response_lengths, args, args.prompt_length, references)
    useful_tokens = sum(args.response_lengths)
    mode_results = {
        "sequential_eager": _mode_result(sequential_latencies, useful_tokens, useful_tokens, sequential_peak),
        "static_batch": _mode_result(static_latencies, useful_tokens,
                                      static_computed_tokens(args.response_lengths, capacity), static_peak),
        "continuous_batch": _mode_result(continuous_latencies, useful_tokens, useful_tokens, continuous_peak),
    }
    environment = {"torch": torch.__version__, "cuda": torch.version.cuda,
                   "transformers": __import__("transformers").__version__,
                   "deepspeed": deepspeed.__version__, "gpu": torch.cuda.get_device_name(device)}
    result = _build_result(args, environment, mode_results, useful_tokens)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote benchmark results to {output_path}")
    return result


if __name__ == "__main__":
    run(_build_parser().parse_args())
