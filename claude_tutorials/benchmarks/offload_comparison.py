#!/usr/bin/env python3
"""
DeepSpeed Offload Strategy Comparison Benchmark

This script compares different offloading strategies:
- No offload (GPU only)
- CPU offload (optimizer)
- CPU offload (optimizer + parameters)
- NVMe offload (parameters)

Usage:
    python offload_comparison.py --model meta-llama/Llama-2-7b-hf --batch-size 4

Output:
    - results/offload_comparison_<timestamp>.json
    - results/offload_comparison_<timestamp>.csv
"""

import argparse
import json
import time
import os
from datetime import datetime
from typing import Dict, List
import csv

import torch
import deepspeed
from transformers import AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import numpy as np


class DummyDataset(Dataset):
    """Dummy dataset for benchmarking."""

    def __init__(self, size: int, seq_length: int, vocab_size: int = 50000):
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        labels = input_ids.clone()
        attention_mask = torch.ones(self.seq_length, dtype=torch.long)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


def get_memory_stats():
    """Get GPU and CPU memory stats."""
    gpu_allocated = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    gpu_reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
    gpu_peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    try:
        import psutil
        cpu_memory = psutil.virtual_memory().used / 1e9
        cpu_percent = psutil.virtual_memory().percent
    except ImportError:
        cpu_memory = 0
        cpu_percent = 0

    return {
        "gpu_allocated_gb": gpu_allocated,
        "gpu_reserved_gb": gpu_reserved,
        "gpu_peak_gb": gpu_peak,
        "cpu_used_gb": cpu_memory,
        "cpu_percent": cpu_percent
    }


def create_offload_config(
    strategy: str,
    batch_size: int,
    nvme_path: str = None
) -> Dict:
    """
    Create DeepSpeed config for different offload strategies.

    Strategies:
    - "none": No offload (ZeRO-3, GPU only)
    - "cpu_optimizer": ZeRO-3 + CPU optimizer offload
    - "cpu_full": ZeRO-3 + CPU optimizer + parameter offload
    - "nvme": ZeRO-3 + NVMe parameter offload (if path provided)
    """

    config = {
        "train_batch_size": batch_size,
        "train_micro_batch_size_per_gpu": batch_size,
        "steps_per_print": 10,
        "gradient_clipping": 1.0,
        "bf16": {"enabled": torch.cuda.is_bf16_supported()},
        "fp16": {"enabled": not torch.cuda.is_bf16_supported()},
        "zero_optimization": {
            "stage": 3,
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "overlap_comm": True,
            "contiguous_gradients": True,
        }
    }

    if strategy == "cpu_optimizer":
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True
        }

    elif strategy == "cpu_full":
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True
        }
        config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True
        }

    elif strategy == "nvme" and nvme_path:
        config["zero_optimization"]["offload_param"] = {
            "device": "nvme",
            "nvme_path": nvme_path,
            "pin_memory": True,
            "buffer_count": 5,
            "buffer_size": 1e8
        }
        config["aio"] = {
            "block_size": 1048576,
            "queue_depth": 8,
            "thread_count": 1,
            "single_submit": False,
            "overlap_events": True
        }

    return config


def benchmark_offload_strategy(
    model_name: str,
    strategy: str,
    batch_size: int,
    seq_length: int,
    num_steps: int,
    warmup_steps: int = 5,
    nvme_path: str = None
) -> Dict:
    """Benchmark a specific offload strategy."""

    print(f"\n{'='*60}")
    print(f"Benchmarking: {strategy.upper().replace('_', ' ')}")
    print(f"{'='*60}")

    # Reset stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    try:
        # Load model
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        # Create config
        ds_config = create_offload_config(strategy, batch_size, nvme_path)

        # Create dataset
        dataset = DummyDataset(size=1000, seq_length=seq_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Initialize DeepSpeed
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config
        )

        mem_after_init = get_memory_stats()
        print(f"Memory after init: GPU={mem_after_init['gpu_allocated_gb']:.2f}GB, "
              f"CPU={mem_after_init['cpu_used_gb']:.2f}GB")

        # Training loop
        model_engine.train()
        step_times = []
        losses = []

        print(f"\nRunning {warmup_steps} warmup + {num_steps} benchmark steps...")

        for step, batch in enumerate(dataloader):
            if step >= warmup_steps + num_steps:
                break

            batch = {k: v.to(model_engine.device) for k, v in batch.items()}

            if step >= warmup_steps:
                torch.cuda.synchronize()
                step_start = time.time()

            # Forward + Backward + Step
            outputs = model_engine(**batch)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()

            if step >= warmup_steps:
                torch.cuda.synchronize()
                step_end = time.time()
                step_times.append(step_end - step_start)
                losses.append(loss.item())

                if (step - warmup_steps) % 5 == 0:
                    print(f"Step {step - warmup_steps}/{num_steps}: "
                          f"Loss={loss.item():.4f}, Time={step_times[-1]*1000:.0f}ms")

        # Get final stats
        final_mem = get_memory_stats()

        # Calculate metrics
        avg_time = np.mean(step_times)
        std_time = np.std(step_times)
        throughput = batch_size * seq_length / avg_time

        results = {
            "strategy": strategy,
            "model": model_name,
            "total_params": total_params,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "num_steps": num_steps,
            "avg_step_time_ms": avg_time * 1000,
            "std_step_time_ms": std_time * 1000,
            "throughput_tokens_per_sec": throughput,
            "avg_loss": np.mean(losses),
            "gpu_peak_gb": final_mem["gpu_peak_gb"],
            "gpu_allocated_gb": final_mem["gpu_allocated_gb"],
            "cpu_used_gb": final_mem["cpu_used_gb"],
            "cpu_percent": final_mem["cpu_percent"],
            "success": True,
            "error": None
        }

        print(f"\n{'='*60}")
        print(f"Results for {strategy}:")
        print(f"  Avg step time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"  Throughput: {throughput:.0f} tokens/sec")
        print(f"  GPU peak: {final_mem['gpu_peak_gb']:.2f} GB")
        print(f"  CPU used: {final_mem['cpu_used_gb']:.2f} GB ({final_mem['cpu_percent']:.1f}%)")
        print(f"{'='*60}\n")

        # Cleanup
        del model_engine, optimizer, model
        torch.cuda.empty_cache()

        return results

    except RuntimeError as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower():
            print(f"\n❌ {strategy}: OUT OF MEMORY")
            return {
                "strategy": strategy,
                "success": False,
                "error": "OOM",
                "error_details": error_msg
            }
        else:
            print(f"\n❌ {strategy}: ERROR - {error_msg}")
            return {
                "strategy": strategy,
                "success": False,
                "error": "RuntimeError",
                "error_details": error_msg
            }
    except Exception as e:
        print(f"\n❌ {strategy}: ERROR - {str(e)}")
        return {
            "strategy": strategy,
            "success": False,
            "error": type(e).__name__,
            "error_details": str(e)
        }


def save_results(results: List[Dict], output_dir: str = "results"):
    """Save results to JSON and CSV."""

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = os.path.join(output_dir, f"offload_comparison_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to {json_path}")

    # CSV
    csv_path = os.path.join(output_dir, f"offload_comparison_{timestamp}.csv")
    successful = [r for r in results if r.get("success", False)]

    if successful:
        fieldnames = ["strategy", "model", "total_params", "batch_size",
                     "avg_step_time_ms", "throughput_tokens_per_sec",
                     "gpu_peak_gb", "cpu_used_gb", "cpu_percent"]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in successful:
                row = {k: result.get(k, "N/A") for k in fieldnames}
                writer.writerow(row)
        print(f"✅ CSV saved to {csv_path}")

    # Print summary
    print("\n" + "="*90)
    print("OFFLOAD STRATEGY COMPARISON")
    print("="*90)
    print(f"{'Strategy':<20} {'Status':<12} {'GPU Mem':<12} {'CPU Mem':<12} {'Time (ms)':<12} {'Throughput'}")
    print("-"*90)

    for result in results:
        strategy = result['strategy'].replace('_', ' ').title()
        if result.get('success'):
            status = "✅ Success"
            gpu_mem = f"{result['gpu_peak_gb']:.2f} GB"
            cpu_mem = f"{result['cpu_used_gb']:.1f} GB"
            time_ms = f"{result['avg_step_time_ms']:.1f}"
            throughput = f"{result['throughput_tokens_per_sec']:.0f} tok/s"
        else:
            status = f"❌ {result['error']}"
            gpu_mem = cpu_mem = time_ms = throughput = "N/A"

        print(f"{strategy:<20} {status:<12} {gpu_mem:<12} {cpu_mem:<12} {time_ms:<12} {throughput}")

    print("="*90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark DeepSpeed offload strategies")
    parser.add_argument("--model", type=str, default="gpt2",
                       help="Model name or path")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size per GPU")
    parser.add_argument("--seq-length", type=int, default=512,
                       help="Sequence length")
    parser.add_argument("--num-steps", type=int, default=20,
                       help="Number of benchmark steps")
    parser.add_argument("--warmup-steps", type=int, default=5,
                       help="Number of warmup steps")
    parser.add_argument("--strategies", type=str, nargs="+",
                       default=["none", "cpu_optimizer", "cpu_full"],
                       help="Strategies to test")
    parser.add_argument("--nvme-path", type=str, default=None,
                       help="Path for NVMe offload (enables nvme strategy)")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank")

    args = parser.parse_args()

    # Add nvme to strategies if path provided
    if args.nvme_path and "nvme" not in args.strategies:
        args.strategies.append("nvme")

    print("\n" + "="*80)
    print("DeepSpeed Offload Strategy Comparison")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Strategies: {args.strategies}")
    print("="*80 + "\n")

    results = []

    for strategy in args.strategies:
        result = benchmark_offload_strategy(
            model_name=args.model,
            strategy=strategy,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            num_steps=args.num_steps,
            warmup_steps=args.warmup_steps,
            nvme_path=args.nvme_path
        )
        results.append(result)
        time.sleep(2)

    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
