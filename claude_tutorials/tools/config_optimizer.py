#!/usr/bin/env python3
"""
DeepSpeed Configuration Optimizer

Automatically tune DeepSpeed configurations by running benchmarks
and finding optimal settings for your model and hardware.

Usage:
    python config_optimizer.py --model your_model.py --num-gpus 8

Author: DeepSpeed Community
License: Apache 2.0
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import itertools


@dataclass
class BenchmarkResult:
    """Store results from a single benchmark run."""
    config: Dict[str, Any]
    success: bool
    avg_step_time: float
    throughput: float
    peak_memory_gb: float
    error_message: Optional[str] = None


class ConfigOptimizer:
    """Optimize DeepSpeed configuration through automated benchmarking."""

    def __init__(
        self,
        model_script: str,
        num_gpus: int,
        num_steps: int = 20,
        warmup_steps: int = 5,
        output_dir: str = "optimization_results"
    ):
        self.model_script = model_script
        self.num_gpus = num_gpus
        self.num_steps = num_steps
        self.warmup_steps = warmup_steps
        self.output_dir = output_dir
        self.results: List[BenchmarkResult] = []

        os.makedirs(output_dir, exist_ok=True)

    def generate_candidate_configs(self, base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate candidate configurations to test."""
        candidates = []

        # ZeRO stages to test
        zero_stages = [0, 1, 2, 3]

        # Batch size variations
        base_batch = base_config.get("train_micro_batch_size_per_gpu", 4)
        batch_sizes = [base_batch // 2, base_batch, base_batch * 2]

        # Communication overlap
        overlap_options = [True, False]

        # Bucket sizes
        bucket_sizes = [int(1e8), int(5e8), int(1e9)]

        # Generate combinations
        print(f"Generating candidate configurations...")
        for zero_stage in zero_stages:
            for batch_size in batch_sizes:
                for overlap in overlap_options:
                    for bucket_size in bucket_sizes:
                        config = self._create_config(
                            base_config,
                            zero_stage=zero_stage,
                            batch_size=batch_size,
                            overlap_comm=overlap,
                            bucket_size=bucket_size
                        )
                        candidates.append(config)

        print(f"Generated {len(candidates)} candidate configurations")
        return candidates

    def _create_config(
        self,
        base_config: Dict[str, Any],
        zero_stage: int,
        batch_size: int,
        overlap_comm: bool,
        bucket_size: int
    ) -> Dict[str, Any]:
        """Create a configuration with specific parameters."""
        config = base_config.copy()

        config["train_micro_batch_size_per_gpu"] = batch_size
        config["train_batch_size"] = batch_size * self.num_gpus * config.get("gradient_accumulation_steps", 1)

        # ZeRO configuration
        zero_config = {"stage": zero_stage}

        if zero_stage >= 2:
            zero_config["contiguous_gradients"] = True
            zero_config["overlap_comm"] = overlap_comm
            zero_config["reduce_bucket_size"] = bucket_size
            zero_config["allgather_bucket_size"] = bucket_size

        if zero_stage == 3:
            zero_config["stage3_prefetch_bucket_size"] = bucket_size
            zero_config["stage3_param_persistence_threshold"] = "auto"
            zero_config["stage3_max_live_parameters"] = int(1e9)
            zero_config["stage3_max_reuse_distance"] = int(1e9)

        config["zero_optimization"] = zero_config

        return config

    def benchmark_config(self, config: Dict[str, Any], config_name: str) -> BenchmarkResult:
        """Run benchmark with given configuration."""
        print(f"\n{'='*60}")
        print(f"Benchmarking: {config_name}")
        print(f"{'='*60}")
        print(f"  ZeRO Stage: {config['zero_optimization']['stage']}")
        print(f"  Batch Size: {config['train_micro_batch_size_per_gpu']}")
        print(f"  Overlap Comm: {config['zero_optimization'].get('overlap_comm', False)}")

        # Save config temporarily
        config_path = os.path.join(self.output_dir, f"{config_name}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Run benchmark
        try:
            result = self._run_benchmark(config_path, config_name)
            print(f"  ✓ Success! Avg step time: {result.avg_step_time*1000:.2f}ms, "
                  f"Memory: {result.peak_memory_gb:.2f}GB")
            return result
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            return BenchmarkResult(
                config=config,
                success=False,
                avg_step_time=float('inf'),
                throughput=0.0,
                peak_memory_gb=0.0,
                error_message=str(e)
            )

    def _run_benchmark(self, config_path: str, config_name: str) -> BenchmarkResult:
        """Execute the benchmark and parse results."""
        # Construct DeepSpeed command
        cmd = [
            "deepspeed",
            f"--num_gpus={self.num_gpus}",
            self.model_script,
            f"--deepspeed_config={config_path}",
            f"--num_steps={self.num_steps}",
            f"--warmup_steps={self.warmup_steps}",
            "--benchmark"
        ]

        # Run benchmark
        print(f"  Running: {' '.join(cmd)}")
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                check=True
            )
            elapsed_time = time.time() - start_time

            # Parse output
            output = result.stdout + result.stderr
            metrics = self._parse_benchmark_output(output)

            with open(config_path, 'r') as f:
                config = json.load(f)

            return BenchmarkResult(
                config=config,
                success=True,
                avg_step_time=metrics['avg_step_time'],
                throughput=metrics['throughput'],
                peak_memory_gb=metrics['peak_memory']
            )

        except subprocess.TimeoutExpired:
            raise Exception("Benchmark timed out")
        except subprocess.CalledProcessError as e:
            raise Exception(f"Benchmark failed with return code {e.returncode}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")

    def _parse_benchmark_output(self, output: str) -> Dict[str, float]:
        """Parse benchmark output to extract metrics."""
        metrics = {
            'avg_step_time': float('inf'),
            'throughput': 0.0,
            'peak_memory': 0.0
        }

        # Parse metrics from output
        for line in output.split('\n'):
            if "Average step time:" in line:
                try:
                    # Extract: "Average step time: 0.45s"
                    time_str = line.split(":")[-1].strip().replace('s', '').replace('ms', '')
                    metrics['avg_step_time'] = float(time_str)
                    if 'ms' in line:
                        metrics['avg_step_time'] /= 1000
                except:
                    pass

            elif "Throughput:" in line:
                try:
                    # Extract: "Throughput: 1234.5 tokens/sec"
                    throughput_str = line.split(":")[1].strip().split()[0]
                    metrics['throughput'] = float(throughput_str)
                except:
                    pass

            elif "Peak memory:" in line or "peak_memory" in line:
                try:
                    # Extract: "Peak memory: 45.2 GB"
                    memory_str = line.split(":")[1].strip().replace('GB', '').replace('GiB', '').split()[0]
                    metrics['peak_memory'] = float(memory_str)
                except:
                    pass

        return metrics

    def optimize(
        self,
        base_config: Dict[str, Any],
        max_configs: int = 20,
        goal: str = "balanced"
    ) -> Tuple[Dict[str, Any], BenchmarkResult]:
        """
        Run optimization to find best configuration.

        Args:
            base_config: Base configuration to start from
            max_configs: Maximum number of configurations to test
            goal: Optimization goal ("speed", "memory", or "balanced")

        Returns:
            Tuple of (best_config, best_result)
        """
        print(f"\n{'='*60}")
        print("Starting Configuration Optimization")
        print(f"{'='*60}")
        print(f"Goal: {goal}")
        print(f"Max configurations to test: {max_configs}")

        # Generate candidates
        candidates = self.generate_candidate_configs(base_config)

        # Limit number of tests
        if len(candidates) > max_configs:
            print(f"Limiting tests to {max_configs} configurations")
            # Prioritize diversity: test different ZeRO stages
            selected = []
            for stage in [0, 1, 2, 3]:
                stage_configs = [c for c in candidates if c["zero_optimization"]["stage"] == stage]
                selected.extend(stage_configs[:max_configs//4])
            candidates = selected[:max_configs]

        # Benchmark each candidate
        for i, config in enumerate(candidates):
            config_name = f"config_{i:03d}_stage{config['zero_optimization']['stage']}_bs{config['train_micro_batch_size_per_gpu']}"
            result = self.benchmark_config(config, config_name)
            self.results.append(result)

            # Save intermediate results
            self._save_results()

        # Find best configuration
        best_config, best_result = self._select_best(goal)

        print(f"\n{'='*60}")
        print("Optimization Complete!")
        print(f"{'='*60}")
        print(f"\nBest Configuration:")
        print(f"  ZeRO Stage: {best_config['zero_optimization']['stage']}")
        print(f"  Batch Size: {best_config['train_micro_batch_size_per_gpu']}")
        print(f"  Overlap Comm: {best_config['zero_optimization'].get('overlap_comm', False)}")
        print(f"  Bucket Size: {best_config['zero_optimization'].get('reduce_bucket_size', 'N/A')}")
        print(f"\nPerformance:")
        print(f"  Avg Step Time: {best_result.avg_step_time*1000:.2f}ms")
        print(f"  Throughput: {best_result.throughput:.1f} tokens/sec")
        print(f"  Peak Memory: {best_result.peak_memory_gb:.2f}GB")

        # Save best config
        best_config_path = os.path.join(self.output_dir, "best_config.json")
        with open(best_config_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        print(f"\nBest configuration saved to: {best_config_path}")

        return best_config, best_result

    def _select_best(self, goal: str) -> Tuple[Dict[str, Any], BenchmarkResult]:
        """Select best configuration based on goal."""
        # Filter successful runs
        successful = [r for r in self.results if r.success]

        if not successful:
            raise Exception("No successful benchmark runs!")

        if goal == "speed":
            # Minimize step time
            best = min(successful, key=lambda r: r.avg_step_time)
        elif goal == "memory":
            # Minimize memory usage (among successful runs)
            best = min(successful, key=lambda r: r.peak_memory_gb)
        else:  # balanced
            # Balance speed and memory
            # Normalize metrics and compute weighted score
            max_time = max(r.avg_step_time for r in successful)
            max_mem = max(r.peak_memory_gb for r in successful)

            def score(r):
                time_score = r.avg_step_time / max_time
                mem_score = r.peak_memory_gb / max_mem
                return 0.6 * time_score + 0.4 * mem_score  # Weight speed more

            best = min(successful, key=score)

        return best.config, best

    def _save_results(self):
        """Save benchmark results to file."""
        results_path = os.path.join(self.output_dir, "optimization_results.json")

        results_data = []
        for r in self.results:
            results_data.append({
                "config": r.config,
                "success": r.success,
                "avg_step_time_ms": r.avg_step_time * 1000 if r.success else None,
                "throughput_tokens_per_sec": r.throughput if r.success else None,
                "peak_memory_gb": r.peak_memory_gb if r.success else None,
                "error": r.error_message
            })

        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)


def load_base_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load base configuration or create default."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    # Default configuration
    return {
        "train_batch_size": 128,
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4
            }
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Optimize DeepSpeed configuration through automated benchmarking"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model training script"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=True,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--base-config",
        type=str,
        help="Base configuration file (optional)"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="Number of steps per benchmark (default: 20)"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of warmup steps (default: 5)"
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=20,
        help="Maximum number of configurations to test (default: 20)"
    )
    parser.add_argument(
        "--goal",
        type=str,
        choices=["speed", "memory", "balanced"],
        default="balanced",
        help="Optimization goal (default: balanced)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="optimization_results",
        help="Output directory for results (default: optimization_results)"
    )

    args = parser.parse_args()

    # Validate model script exists
    if not os.path.exists(args.model):
        print(f"Error: Model script not found: {args.model}")
        sys.exit(1)

    # Load base configuration
    base_config = load_base_config(args.base_config)

    # Create optimizer
    optimizer = ConfigOptimizer(
        model_script=args.model,
        num_gpus=args.num_gpus,
        num_steps=args.num_steps,
        warmup_steps=args.warmup_steps,
        output_dir=args.output_dir
    )

    # Run optimization
    try:
        best_config, best_result = optimizer.optimize(
            base_config=base_config,
            max_configs=args.max_configs,
            goal=args.goal
        )

        print("\n" + "="*60)
        print("Optimization completed successfully!")
        print("="*60)
        print(f"\nResults saved to: {args.output_dir}/")
        print(f"Best config: {args.output_dir}/best_config.json")
        print(f"All results: {args.output_dir}/optimization_results.json")

    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nOptimization failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
