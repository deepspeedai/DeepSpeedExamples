#!/usr/bin/env python3
"""
Framework Comparison Benchmark Tool

Compares DeepSpeed, PyTorch FSDP, and Hugging Face Accelerate across different
model sizes and configurations. Measures throughput, memory usage, and scalability.

Usage:
    python framework_comparison.py --model gpt2 --frameworks deepspeed fsdp accelerate
    python framework_comparison.py --model llama-7b --batch-size 4 --seq-length 2048
    python framework_comparison.py --quick-benchmark  # Run all standard benchmarks
    python framework_comparison.py --compare-configs  # Compare different ZeRO stages

Example Output:
    ┌────────────────────────────────────────────────────────────────┐
    │ Framework Comparison: GPT-2 (1.5B params)                      │
    ├────────────┬─────────────┬────────────┬──────────────┬─────────┤
    │ Framework  │ Throughput  │ Memory/GPU │ Scaling Eff  │ Setup   │
    ├────────────┼─────────────┼────────────┼──────────────┼─────────┤
    │ DeepSpeed  │ 2,400 tok/s │ 28GB       │ 95%          │ Medium  │
    │ FSDP       │ 2,350 tok/s │ 30GB       │ 92%          │ Low     │
    │ Accelerate │ 2,400 tok/s │ 28GB       │ 95%          │ Very Low│
    └────────────┴─────────────┴────────────┴──────────────┴─────────┘

Requirements:
    pip install torch transformers deepspeed accelerate tabulate psutil
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

try:
    import torch
    import psutil
    from tabulate import tabulate
except ImportError as e:
    print(f"Error: Missing required package. Install with:")
    print(f"  pip install torch transformers deepspeed accelerate tabulate psutil")
    sys.exit(1)


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    framework: str  # deepspeed, fsdp, accelerate, ddp
    model_name: str  # gpt2, gpt2-medium, gpt2-large, llama-7b, etc.
    batch_size: int
    seq_length: int
    num_gpus: int
    zero_stage: Optional[int] = None  # For DeepSpeed
    use_offload: bool = False
    use_fp16: bool = True
    gradient_accumulation_steps: int = 1


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    config: BenchmarkConfig
    throughput_tokens_per_sec: float
    memory_per_gpu_gb: float
    peak_memory_gb: float
    time_per_step_ms: float
    scaling_efficiency: float  # vs single GPU baseline
    setup_complexity: str  # Low, Medium, High
    success: bool
    error_message: Optional[str] = None


class FrameworkBenchmark:
    """Base class for framework benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_model(self):
        """Setup model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {self.config.model_name}")

        # Map friendly names to HF model IDs
        model_map = {
            "gpt2": "gpt2",
            "gpt2-medium": "gpt2-medium",
            "gpt2-large": "gpt2-large",
            "gpt2-xl": "gpt2-xl",
            "llama-7b": "meta-llama/Llama-2-7b-hf",  # Requires access
            "opt-1.3b": "facebook/opt-1.3b",
            "opt-6.7b": "facebook/opt-6.7b",
        }

        model_id = model_map.get(self.config.model_name, self.config.model_name)

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32,
                low_cpu_mem_usage=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def generate_dummy_data(self, num_batches=10):
        """Generate dummy input data for benchmarking."""
        input_ids = torch.randint(
            0, self.tokenizer.vocab_size,
            (self.config.batch_size, self.config.seq_length)
        )
        attention_mask = torch.ones_like(input_ids)

        return [
            {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}
            for _ in range(num_batches)
        ]

    def measure_memory(self):
        """Measure current GPU memory usage."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / 1024**3  # GB
        return 0.0

    def run(self) -> BenchmarkResult:
        """Run benchmark and return results."""
        raise NotImplementedError("Subclasses must implement run()")


class DeepSpeedBenchmark(FrameworkBenchmark):
    """Benchmark using DeepSpeed."""

    def run(self) -> BenchmarkResult:
        try:
            import deepspeed
        except ImportError:
            return BenchmarkResult(
                config=self.config,
                throughput_tokens_per_sec=0.0,
                memory_per_gpu_gb=0.0,
                peak_memory_gb=0.0,
                time_per_step_ms=0.0,
                scaling_efficiency=0.0,
                setup_complexity="Medium",
                success=False,
                error_message="DeepSpeed not installed"
            )

        if not self.setup_model():
            return BenchmarkResult(
                config=self.config,
                throughput_tokens_per_sec=0.0,
                memory_per_gpu_gb=0.0,
                peak_memory_gb=0.0,
                time_per_step_ms=0.0,
                scaling_efficiency=0.0,
                setup_complexity="Medium",
                success=False,
                error_message="Model loading failed"
            )

        # DeepSpeed config
        ds_config = {
            "train_batch_size": self.config.batch_size * self.config.num_gpus,
            "train_micro_batch_size_per_gpu": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "fp16": {"enabled": self.config.use_fp16},
            "zero_optimization": {
                "stage": self.config.zero_stage or 2,
                "overlap_comm": True,
            },
            "steps_per_print": 999999,
        }

        if self.config.use_offload and self.config.zero_stage == 3:
            ds_config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True
            }

        # Initialize DeepSpeed
        try:
            model_engine, optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                config=ds_config
            )
        except Exception as e:
            return BenchmarkResult(
                config=self.config,
                throughput_tokens_per_sec=0.0,
                memory_per_gpu_gb=0.0,
                peak_memory_gb=0.0,
                time_per_step_ms=0.0,
                scaling_efficiency=0.0,
                setup_complexity="Medium",
                success=False,
                error_message=f"DeepSpeed init failed: {e}"
            )

        # Generate data
        dummy_data = self.generate_dummy_data()

        # Warmup
        print("Warming up...")
        for i, batch in enumerate(dummy_data[:3]):
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            outputs = model_engine(**batch)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()

        # Benchmark
        print("Benchmarking...")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_time = time.time()

        num_steps = 0
        total_tokens = 0

        for batch in dummy_data:
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}

            outputs = model_engine(**batch)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()

            num_steps += 1
            total_tokens += self.config.batch_size * self.config.seq_length * self.config.num_gpus

        torch.cuda.synchronize()
        end_time = time.time()

        elapsed_time = end_time - start_time
        throughput = total_tokens / elapsed_time
        time_per_step = (elapsed_time / num_steps) * 1000  # ms
        memory_used = self.measure_memory()

        return BenchmarkResult(
            config=self.config,
            throughput_tokens_per_sec=throughput,
            memory_per_gpu_gb=memory_used / self.config.num_gpus,
            peak_memory_gb=memory_used,
            time_per_step_ms=time_per_step,
            scaling_efficiency=100.0,  # Placeholder
            setup_complexity="Medium",
            success=True
        )


class FSDPBenchmark(FrameworkBenchmark):
    """Benchmark using PyTorch FSDP."""

    def run(self) -> BenchmarkResult:
        try:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                MixedPrecision,
                ShardingStrategy
            )
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        except ImportError:
            return BenchmarkResult(
                config=self.config,
                throughput_tokens_per_sec=0.0,
                memory_per_gpu_gb=0.0,
                peak_memory_gb=0.0,
                time_per_step_ms=0.0,
                scaling_efficiency=0.0,
                setup_complexity="Low",
                success=False,
                error_message="FSDP not available (PyTorch 2.0+ required)"
            )

        if not self.setup_model():
            return BenchmarkResult(
                config=self.config,
                throughput_tokens_per_sec=0.0,
                memory_per_gpu_gb=0.0,
                peak_memory_gb=0.0,
                time_per_step_ms=0.0,
                scaling_efficiency=0.0,
                setup_complexity="Low",
                success=False,
                error_message="Model loading failed"
            )

        # FSDP config
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16 if self.config.use_fp16 else torch.float32,
            reduce_dtype=torch.float16 if self.config.use_fp16 else torch.float32,
            buffer_dtype=torch.float16 if self.config.use_fp16 else torch.float32,
        )

        # Wrap with FSDP
        try:
            # Assume GPT-2 style model
            from transformers.models.gpt2.modeling_gpt2 import GPT2Block

            auto_wrap_policy = transformer_auto_wrap_policy(
                transformer_layer_cls={GPT2Block}
            )

            self.model = FSDP(
                self.model,
                mixed_precision=mixed_precision_policy,
                auto_wrap_policy=auto_wrap_policy,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=torch.cuda.current_device(),
            )
        except Exception as e:
            return BenchmarkResult(
                config=self.config,
                throughput_tokens_per_sec=0.0,
                memory_per_gpu_gb=0.0,
                peak_memory_gb=0.0,
                time_per_step_ms=0.0,
                scaling_efficiency=0.0,
                setup_complexity="Low",
                success=False,
                error_message=f"FSDP wrapping failed: {e}"
            )

        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

        # Generate data
        dummy_data = self.generate_dummy_data()

        # Warmup
        print("Warming up...")
        for batch in dummy_data[:3]:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        # Benchmark
        print("Benchmarking...")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_time = time.time()

        num_steps = 0
        total_tokens = 0

        for batch in dummy_data:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            num_steps += 1
            total_tokens += self.config.batch_size * self.config.seq_length * self.config.num_gpus

        torch.cuda.synchronize()
        end_time = time.time()

        elapsed_time = end_time - start_time
        throughput = total_tokens / elapsed_time
        time_per_step = (elapsed_time / num_steps) * 1000  # ms
        memory_used = self.measure_memory()

        return BenchmarkResult(
            config=self.config,
            throughput_tokens_per_sec=throughput,
            memory_per_gpu_gb=memory_used / self.config.num_gpus,
            peak_memory_gb=memory_used,
            time_per_step_ms=time_per_step,
            scaling_efficiency=100.0,  # Placeholder
            setup_complexity="Low",
            success=True
        )


class AccelerateBenchmark(FrameworkBenchmark):
    """Benchmark using Hugging Face Accelerate."""

    def run(self) -> BenchmarkResult:
        try:
            from accelerate import Accelerator
        except ImportError:
            return BenchmarkResult(
                config=self.config,
                throughput_tokens_per_sec=0.0,
                memory_per_gpu_gb=0.0,
                peak_memory_gb=0.0,
                time_per_step_ms=0.0,
                scaling_efficiency=0.0,
                setup_complexity="Very Low",
                success=False,
                error_message="Accelerate not installed"
            )

        if not self.setup_model():
            return BenchmarkResult(
                config=self.config,
                throughput_tokens_per_sec=0.0,
                memory_per_gpu_gb=0.0,
                peak_memory_gb=0.0,
                time_per_step_ms=0.0,
                scaling_efficiency=0.0,
                setup_complexity="Very Low",
                success=False,
                error_message="Model loading failed"
            )

        # Initialize Accelerator
        accelerator = Accelerator(
            mixed_precision="fp16" if self.config.use_fp16 else "no",
            gradient_accumulation_steps=self.config.gradient_accumulation_steps
        )

        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

        # Prepare
        self.model, optimizer = accelerator.prepare(self.model, optimizer)

        # Generate data
        dummy_data = self.generate_dummy_data()

        # Warmup
        print("Warming up...")
        for batch in dummy_data[:3]:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()

        # Benchmark
        print("Benchmarking...")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_time = time.time()

        num_steps = 0
        total_tokens = 0

        for batch in dummy_data:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()

            num_steps += 1
            total_tokens += self.config.batch_size * self.config.seq_length * self.config.num_gpus

        torch.cuda.synchronize()
        end_time = time.time()

        elapsed_time = end_time - start_time
        throughput = total_tokens / elapsed_time
        time_per_step = (elapsed_time / num_steps) * 1000  # ms
        memory_used = self.measure_memory()

        return BenchmarkResult(
            config=self.config,
            throughput_tokens_per_sec=throughput,
            memory_per_gpu_gb=memory_used / self.config.num_gpus,
            peak_memory_gb=memory_used,
            time_per_step_ms=time_per_step,
            scaling_efficiency=100.0,  # Placeholder
            setup_complexity="Very Low",
            success=True
        )


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run a single benchmark with the given configuration."""
    print(f"\n{'='*70}")
    print(f"Running {config.framework.upper()} benchmark:")
    print(f"  Model: {config.model_name}")
    print(f"  Batch size: {config.batch_size}, Sequence length: {config.seq_length}")
    print(f"  GPUs: {config.num_gpus}, FP16: {config.use_fp16}")
    if config.framework == "deepspeed":
        print(f"  ZeRO stage: {config.zero_stage}, Offload: {config.use_offload}")
    print(f"{'='*70}")

    benchmark_class = {
        "deepspeed": DeepSpeedBenchmark,
        "fsdp": FSDPBenchmark,
        "accelerate": AccelerateBenchmark,
    }.get(config.framework)

    if benchmark_class is None:
        return BenchmarkResult(
            config=config,
            throughput_tokens_per_sec=0.0,
            memory_per_gpu_gb=0.0,
            peak_memory_gb=0.0,
            time_per_step_ms=0.0,
            scaling_efficiency=0.0,
            setup_complexity="Unknown",
            success=False,
            error_message=f"Unknown framework: {config.framework}"
        )

    try:
        benchmark = benchmark_class(config)
        return benchmark.run()
    except Exception as e:
        print(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        return BenchmarkResult(
            config=config,
            throughput_tokens_per_sec=0.0,
            memory_per_gpu_gb=0.0,
            peak_memory_gb=0.0,
            time_per_step_ms=0.0,
            scaling_efficiency=0.0,
            setup_complexity="Unknown",
            success=False,
            error_message=str(e)
        )


def print_results_table(results: List[BenchmarkResult]):
    """Print benchmark results in a formatted table."""
    headers = [
        "Framework",
        "Model",
        "Throughput\n(tokens/s)",
        "Memory/GPU\n(GB)",
        "Time/Step\n(ms)",
        "Setup",
        "Status"
    ]

    rows = []
    for result in results:
        if result.success:
            rows.append([
                result.config.framework.upper(),
                result.config.model_name,
                f"{result.throughput_tokens_per_sec:,.0f}",
                f"{result.memory_per_gpu_gb:.1f}",
                f"{result.time_per_step_ms:.1f}",
                result.setup_complexity,
                "✓ Success"
            ])
        else:
            rows.append([
                result.config.framework.upper(),
                result.config.model_name,
                "N/A",
                "N/A",
                "N/A",
                result.setup_complexity,
                f"✗ {result.error_message[:30]}"
            ])

    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print("="*80 + "\n")


def save_results(results: List[BenchmarkResult], output_file: str):
    """Save benchmark results to JSON file."""
    data = {
        "results": [
            {
                "config": asdict(r.config),
                "metrics": {
                    "throughput_tokens_per_sec": r.throughput_tokens_per_sec,
                    "memory_per_gpu_gb": r.memory_per_gpu_gb,
                    "peak_memory_gb": r.peak_memory_gb,
                    "time_per_step_ms": r.time_per_step_ms,
                    "scaling_efficiency": r.scaling_efficiency,
                    "setup_complexity": r.setup_complexity,
                },
                "success": r.success,
                "error_message": r.error_message
            }
            for r in results
        ]
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare DeepSpeed, FSDP, and Accelerate frameworks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all frameworks on GPT-2
  python framework_comparison.py --model gpt2 --frameworks deepspeed fsdp accelerate

  # Benchmark larger model
  python framework_comparison.py --model gpt2-xl --batch-size 2 --seq-length 1024

  # Compare DeepSpeed ZeRO stages
  python framework_comparison.py --model gpt2 --framework deepspeed --zero-stages 1 2 3

  # Quick benchmark suite
  python framework_comparison.py --quick-benchmark
        """
    )

    parser.add_argument("--model", default="gpt2", help="Model name (default: gpt2)")
    parser.add_argument("--frameworks", nargs="+", default=["deepspeed", "fsdp", "accelerate"],
                        choices=["deepspeed", "fsdp", "accelerate"],
                        help="Frameworks to benchmark")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 precision")
    parser.add_argument("--zero-stages", nargs="+", type=int, default=[2],
                        help="DeepSpeed ZeRO stages to test (default: 2)")
    parser.add_argument("--use-offload", action="store_true", help="Enable CPU offload (DeepSpeed ZeRO-3)")
    parser.add_argument("--quick-benchmark", action="store_true",
                        help="Run quick benchmark suite across models")
    parser.add_argument("--output", default="benchmark_results.json",
                        help="Output file for results (default: benchmark_results.json)")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Benchmarks will not be accurate.")

    results = []

    if args.quick_benchmark:
        print("Running quick benchmark suite...")
        configs = [
            # GPT-2 small
            BenchmarkConfig("deepspeed", "gpt2", 8, 512, args.num_gpus, zero_stage=2),
            BenchmarkConfig("fsdp", "gpt2", 8, 512, args.num_gpus),
            BenchmarkConfig("accelerate", "gpt2", 8, 512, args.num_gpus),

            # GPT-2 medium
            BenchmarkConfig("deepspeed", "gpt2-medium", 4, 512, args.num_gpus, zero_stage=2),
            BenchmarkConfig("fsdp", "gpt2-medium", 4, 512, args.num_gpus),
            BenchmarkConfig("accelerate", "gpt2-medium", 4, 512, args.num_gpus),
        ]

        for config in configs:
            config.use_fp16 = args.fp16
            result = run_benchmark(config)
            results.append(result)

    else:
        # Run custom benchmarks
        for framework in args.frameworks:
            if framework == "deepspeed":
                for zero_stage in args.zero_stages:
                    config = BenchmarkConfig(
                        framework="deepspeed",
                        model_name=args.model,
                        batch_size=args.batch_size,
                        seq_length=args.seq_length,
                        num_gpus=args.num_gpus,
                        zero_stage=zero_stage,
                        use_offload=args.use_offload and zero_stage == 3,
                        use_fp16=args.fp16
                    )
                    result = run_benchmark(config)
                    results.append(result)
            else:
                config = BenchmarkConfig(
                    framework=framework,
                    model_name=args.model,
                    batch_size=args.batch_size,
                    seq_length=args.seq_length,
                    num_gpus=args.num_gpus,
                    use_fp16=args.fp16
                )
                result = run_benchmark(config)
                results.append(result)

    # Print results
    if results:
        print_results_table(results)
        save_results(results, args.output)

        # Summary
        successful = [r for r in results if r.success]
        if successful:
            best = max(successful, key=lambda r: r.throughput_tokens_per_sec)
            print(f"\n🏆 Best Performance: {best.config.framework.upper()}")
            print(f"   Throughput: {best.throughput_tokens_per_sec:,.0f} tokens/s")
            print(f"   Memory: {best.memory_per_gpu_gb:.1f} GB/GPU\n")


if __name__ == "__main__":
    main()
