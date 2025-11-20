#!/usr/bin/env python3
"""
DeepSpeed ZeRO Stage Comparison Benchmark

This script benchmarks different ZeRO stages (0, 1, 2, 3) on the same model
to measure memory usage, throughput, and training time.

Usage:
    # Single GPU
    python zero_stage_comparison.py --model meta-llama/Llama-2-7b-hf --batch-size 4

    # Multi-GPU
    deepspeed --num_gpus=4 zero_stage_comparison.py --model meta-llama/Llama-2-7b-hf --batch-size 8

Output:
    - Console output with real-time metrics
    - results/zero_comparison_<timestamp>.json with detailed results
    - results/zero_comparison_<timestamp>.csv for easy analysis
"""

import argparse
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Tuple
import csv

import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np


class DummyDataset(Dataset):
    """Dummy dataset for benchmarking (avoids I/O bottleneck)."""

    def __init__(self, size: int, seq_length: int, vocab_size: int = 50000):
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate random tokens
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        labels = input_ids.clone()
        attention_mask = torch.ones(self.seq_length, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        return allocated, reserved, max_allocated
    return 0, 0, 0


def create_deepspeed_config(stage: int, batch_size: int, grad_accum: int = 1) -> Dict:
    """Create DeepSpeed configuration for specified ZeRO stage."""

    config = {
        "train_batch_size": batch_size * grad_accum,
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "steps_per_print": 10,
        "gradient_clipping": 1.0,
        "bf16": {"enabled": torch.cuda.is_bf16_supported()},
        "fp16": {"enabled": not torch.cuda.is_bf16_supported()},
    }

    if stage > 0:
        config["zero_optimization"] = {
            "stage": stage,
        }

        if stage == 1:
            config["zero_optimization"].update({
                "allgather_partitions": True,
                "reduce_scatter": True,
                "overlap_comm": True,
                "contiguous_gradients": True,
            })
        elif stage == 2:
            config["zero_optimization"].update({
                "allgather_partitions": True,
                "reduce_scatter": True,
                "allgather_bucket_size": 5e8,
                "reduce_bucket_size": 5e8,
                "overlap_comm": True,
                "contiguous_gradients": True,
            })
        elif stage == 3:
            config["zero_optimization"].update({
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": "auto",
            })

    return config


def benchmark_zero_stage(
    model_name: str,
    stage: int,
    batch_size: int,
    seq_length: int,
    num_steps: int,
    warmup_steps: int = 5,
    grad_accum: int = 1
) -> Dict:
    """Benchmark a specific ZeRO stage."""

    print(f"\n{'='*60}")
    print(f"Benchmarking ZeRO Stage {stage}")
    print(f"{'='*60}")

    # Reset CUDA stats
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

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Create DeepSpeed config
        ds_config = create_deepspeed_config(stage, batch_size, grad_accum)

        # Create dataset
        dataset = DummyDataset(size=1000, seq_length=seq_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Initialize DeepSpeed
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config
        )

        # Get memory after initialization
        mem_after_init = get_gpu_memory()
        print(f"Memory after init: {mem_after_init[0]:.2f} GB allocated, {mem_after_init[1]:.2f} GB reserved")

        # Training loop
        model_engine.train()
        step_times = []
        losses = []

        print(f"\nStarting benchmark ({warmup_steps} warmup + {num_steps} measured steps)...")

        for step, batch in enumerate(dataloader):
            if step >= warmup_steps + num_steps:
                break

            # Move to device
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}

            # Start timing (after warmup)
            if step >= warmup_steps:
                torch.cuda.synchronize()
                step_start = time.time()

            # Forward pass
            outputs = model_engine(**batch)
            loss = outputs.loss

            # Backward pass
            model_engine.backward(loss)

            # Optimizer step
            model_engine.step()

            # End timing
            if step >= warmup_steps:
                torch.cuda.synchronize()
                step_end = time.time()
                step_times.append(step_end - step_start)
                losses.append(loss.item())

                if (step - warmup_steps) % 10 == 0:
                    print(f"Step {step - warmup_steps}/{num_steps}: "
                          f"Loss={loss.item():.4f}, "
                          f"Time={step_times[-1]*1000:.0f}ms")

        # Get final memory stats
        mem_allocated, mem_reserved, mem_peak = get_gpu_memory()

        # Calculate metrics
        avg_step_time = np.mean(step_times)
        std_step_time = np.std(step_times)
        throughput = batch_size * seq_length / avg_step_time  # tokens/sec
        avg_loss = np.mean(losses)

        results = {
            "stage": stage,
            "model": model_name,
            "total_params": total_params,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "grad_accum": grad_accum,
            "num_steps": num_steps,
            "avg_step_time_ms": avg_step_time * 1000,
            "std_step_time_ms": std_step_time * 1000,
            "throughput_tokens_per_sec": throughput,
            "avg_loss": avg_loss,
            "memory_allocated_gb": mem_allocated,
            "memory_reserved_gb": mem_reserved,
            "memory_peak_gb": mem_peak,
            "success": True,
            "error": None
        }

        print(f"\n{'='*60}")
        print(f"Results for ZeRO Stage {stage}:")
        print(f"  Avg step time: {avg_step_time*1000:.2f} ± {std_step_time*1000:.2f} ms")
        print(f"  Throughput: {throughput:.0f} tokens/sec")
        print(f"  Peak memory: {mem_peak:.2f} GB")
        print(f"  Avg loss: {avg_loss:.4f}")
        print(f"{'='*60}\n")

        # Cleanup
        del model_engine, optimizer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ ZeRO Stage {stage}: OUT OF MEMORY")
            return {
                "stage": stage,
                "success": False,
                "error": "OOM",
                "error_details": str(e)
            }
        else:
            print(f"\n❌ ZeRO Stage {stage}: ERROR - {str(e)}")
            return {
                "stage": stage,
                "success": False,
                "error": "RuntimeError",
                "error_details": str(e)
            }
    except Exception as e:
        print(f"\n❌ ZeRO Stage {stage}: ERROR - {str(e)}")
        return {
            "stage": stage,
            "success": False,
            "error": type(e).__name__,
            "error_details": str(e)
        }


def save_results(results: List[Dict], output_dir: str = "results"):
    """Save benchmark results to JSON and CSV."""

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = os.path.join(output_dir, f"zero_comparison_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to {json_path}")

    # Save CSV (only successful runs)
    csv_path = os.path.join(output_dir, f"zero_comparison_{timestamp}.csv")
    successful_results = [r for r in results if r.get("success", False)]

    if successful_results:
        fieldnames = ["stage", "model", "total_params", "batch_size", "seq_length",
                     "avg_step_time_ms", "throughput_tokens_per_sec",
                     "memory_peak_gb", "avg_loss"]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in successful_results:
                row = {k: result.get(k, "N/A") for k in fieldnames}
                writer.writerow(row)
        print(f"✅ CSV saved to {csv_path}")

    # Print summary table
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Stage':<8} {'Status':<12} {'Memory (GB)':<15} {'Time (ms)':<15} {'Throughput':<15}")
    print("-"*80)

    for result in results:
        stage = result['stage']
        if result.get('success'):
            status = "✅ Success"
            memory = f"{result['memory_peak_gb']:.2f}"
            time_ms = f"{result['avg_step_time_ms']:.1f}"
            throughput = f"{result['throughput_tokens_per_sec']:.0f} tok/s"
        else:
            status = f"❌ {result['error']}"
            memory = "N/A"
            time_ms = "N/A"
            throughput = "N/A"

        print(f"ZeRO-{stage:<3} {status:<12} {memory:<15} {time_ms:<15} {throughput:<15}")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark DeepSpeed ZeRO stages")
    parser.add_argument("--model", type=str, default="gpt2",
                       help="Model name or path (default: gpt2)")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size per GPU (default: 4)")
    parser.add_argument("--seq-length", type=int, default=512,
                       help="Sequence length (default: 512)")
    parser.add_argument("--num-steps", type=int, default=20,
                       help="Number of steps to benchmark (default: 20)")
    parser.add_argument("--warmup-steps", type=int, default=5,
                       help="Number of warmup steps (default: 5)")
    parser.add_argument("--stages", type=int, nargs="+", default=[0, 1, 2, 3],
                       help="ZeRO stages to benchmark (default: 0 1 2 3)")
    parser.add_argument("--grad-accum", type=int, default=1,
                       help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results (default: results)")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank for distributed training")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("DeepSpeed ZeRO Stage Comparison Benchmark")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Stages to test: {args.stages}")
    print(f"Steps: {args.num_steps} (+ {args.warmup_steps} warmup)")
    print("="*80 + "\n")

    results = []

    for stage in args.stages:
        result = benchmark_zero_stage(
            model_name=args.model,
            stage=stage,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            num_steps=args.num_steps,
            warmup_steps=args.warmup_steps,
            grad_accum=args.grad_accum
        )
        results.append(result)

        # Wait a bit between stages
        time.sleep(2)

    # Save results
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            save_results(results, args.output_dir)
    else:
        save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
