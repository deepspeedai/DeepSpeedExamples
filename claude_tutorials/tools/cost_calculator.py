#!/usr/bin/env python3
"""
DeepSpeed Cost Calculator

Calculate and compare training costs across different configurations,
cloud providers, and instance types.

Usage:
    python cost_calculator.py --model-size 7B --steps 100000 --provider aws
    python cost_calculator.py --compare  # Compare all providers

Author: DeepSpeed Community
License: Apache 2.0
"""

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class Provider(Enum):
    """Cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    LAMBDA = "lambda"
    COREWEAVE = "coreweave"


@dataclass
class GPUInstance:
    """GPU instance specification."""
    name: str
    provider: Provider
    gpu_type: str
    num_gpus: int
    gpu_memory_gb: int
    vcpus: int
    ram_gb: int
    price_per_hour: float
    spot_price_per_hour: Optional[float] = None

    def price_per_gpu_hour(self, use_spot: bool = False) -> float:
        """Get price per GPU per hour."""
        price = self.spot_price_per_hour if use_spot and self.spot_price_per_hour else self.price_per_hour
        return price / self.num_gpus


# Pricing data (as of 2024)
INSTANCES = [
    # AWS
    GPUInstance("p4d.24xlarge", Provider.AWS, "A100-40GB", 8, 40, 96, 1152, 32.77, 15.00),
    GPUInstance("p4de.24xlarge", Provider.AWS, "A100-80GB", 8, 80, 96, 1152, 40.96, 18.00),
    GPUInstance("p5.48xlarge", Provider.AWS, "H100-80GB", 8, 80, 192, 2048, 98.32, 45.00),
    GPUInstance("p3.16xlarge", Provider.AWS, "V100-16GB", 8, 16, 64, 488, 24.48, 10.00),
    GPUInstance("g5.48xlarge", Provider.AWS, "A10G-24GB", 8, 24, 192, 768, 16.29, 6.00),

    # GCP
    GPUInstance("a2-highgpu-8g", Provider.GCP, "A100-40GB", 8, 40, 96, 680, 29.39, 12.00),
    GPUInstance("a2-ultragpu-8g", Provider.GCP, "A100-80GB", 8, 80, 96, 1360, 35.73, 14.00),
    GPUInstance("a3-highgpu-8g", Provider.GCP, "H100-80GB", 8, 80, 208, 1872, 74.16, 30.00),

    # Azure
    GPUInstance("ND96asr_v4", Provider.AZURE, "A100-40GB", 8, 40, 96, 900, 27.20, 12.00),
    GPUInstance("ND96amsr_A100_v4", Provider.AZURE, "A100-80GB", 8, 80, 96, 1900, 32.77, 14.00),

    # Lambda Labs
    GPUInstance("gpu_8x_a100_40gb", Provider.LAMBDA, "A100-40GB", 8, 40, 96, 800, 8.80, None),
    GPUInstance("gpu_8x_a100_80gb", Provider.LAMBDA, "A100-80GB", 8, 80, 96, 1400, 10.32, None),

    # CoreWeave
    GPUInstance("gpu_8x_a100_80gb_pcie", Provider.COREWEAVE, "A100-80GB-PCIe", 8, 80, 96, 1000, 16.48, None),
    GPUInstance("gpu_8x_h100_80gb_hbm3", Provider.COREWEAVE, "H100-80GB-HBM3", 8, 80, 192, 2000, 38.08, None),
]


def parse_model_size(size_str: str) -> float:
    """Parse model size string to billions of parameters."""
    size_str = size_str.upper().replace(" ", "")
    if 'B' in size_str:
        return float(size_str.replace('B', ''))
    elif 'M' in size_str:
        return float(size_str.replace('M', '')) / 1000
    else:
        return float(size_str) / 1e9


def estimate_training_time(
    model_size_b: float,
    num_steps: int,
    num_gpus: int,
    gpu_type: str,
    zero_stage: int = 2,
    use_offload: bool = False,
    use_compression: bool = False
) -> float:
    """
    Estimate training time in hours.

    This is a rough estimate based on typical performance characteristics.
    """
    # Base time per step (in seconds) for 7B model on A100
    base_time_per_step = 0.5

    # Scale by model size (roughly linear)
    time_per_step = base_time_per_step * (model_size_b / 7.0)

    # Scale by GPU performance
    gpu_performance = {
        "V100": 0.6,    # Slower
        "A10G": 0.7,
        "A100-40GB": 1.0,  # Baseline
        "A100-80GB": 1.0,
        "H100": 1.5,    # Faster
    }

    # Extract GPU family
    gpu_family = "A100-40GB"
    for key in gpu_performance:
        if key in gpu_type:
            gpu_family = key
            break

    time_per_step /= gpu_performance[gpu_family]

    # Scale by number of GPUs (not perfectly linear)
    scaling_efficiency = {
        1: 1.0,
        2: 0.95,
        4: 0.90,
        8: 0.85,
        16: 0.75,
        32: 0.65,
    }

    gpus = min(num_gpus, 32)
    efficiency = scaling_efficiency.get(gpus, 0.65)
    time_per_step /= (gpus * efficiency)

    # ZeRO stage overhead
    zero_overhead = {
        0: 1.0,
        1: 1.05,
        2: 1.15,
        3: 1.30,
    }
    time_per_step *= zero_overhead.get(zero_stage, 1.15)

    # Offloading overhead
    if use_offload:
        time_per_step *= 1.25

    # Compression benefit (for multi-node)
    if use_compression and num_gpus > 8:
        time_per_step *= 0.70  # 30% speedup

    # Total time
    total_seconds = time_per_step * num_steps
    total_hours = total_seconds / 3600

    return total_hours


def calculate_cost(
    instance: GPUInstance,
    training_hours: float,
    use_spot: bool = False,
    storage_gb: int = 100,
    storage_days: int = 7
) -> Dict[str, float]:
    """Calculate total training cost."""

    # Compute cost
    price_per_hour = instance.spot_price_per_hour if use_spot and instance.spot_price_per_hour else instance.price_per_hour
    compute_cost = training_hours * price_per_hour

    # Storage cost (rough estimate)
    storage_cost_per_gb_month = {
        Provider.AWS: 0.023,
        Provider.GCP: 0.020,
        Provider.AZURE: 0.018,
        Provider.LAMBDA: 0.010,
        Provider.COREWEAVE: 0.015,
    }
    storage_price = storage_cost_per_gb_month.get(instance.provider, 0.020)
    storage_cost = storage_gb * storage_price * (storage_days / 30)

    # Network cost (typically negligible for intra-region)
    network_cost = 0.0

    total_cost = compute_cost + storage_cost + network_cost

    return {
        "compute": compute_cost,
        "storage": storage_cost,
        "network": network_cost,
        "total": total_cost
    }


def find_suitable_instances(
    model_size_b: float,
    zero_stage: int = 2,
    use_offload: bool = False,
    min_gpu_memory: int = 40
) -> List[GPUInstance]:
    """Find instances that can fit the model."""

    # Rough memory estimation
    model_memory_gb = model_size_b * 2  # FP16
    optimizer_memory_gb = model_memory_gb * 2  # Adam states
    gradient_memory_gb = model_memory_gb
    activation_memory_gb = model_memory_gb * 0.5

    total_memory_gb = model_memory_gb + optimizer_memory_gb + gradient_memory_gb + activation_memory_gb

    # ZeRO memory reduction
    zero_reduction = {
        0: 1.0,
        1: 0.75,  # Optimizer partitioned
        2: 0.55,  # Optimizer + gradients partitioned
        3: 0.35,  # Everything partitioned
    }

    memory_per_gpu = (total_memory_gb * zero_reduction.get(zero_stage, 0.55))

    # Offloading reduces GPU memory needs
    if use_offload:
        memory_per_gpu *= 0.6

    # Filter suitable instances
    suitable = []
    for instance in INSTANCES:
        if instance.gpu_memory_gb >= max(min_gpu_memory, memory_per_gpu):
            suitable.append(instance)

    return suitable


def print_cost_comparison(
    model_size_b: float,
    num_steps: int,
    zero_stage: int,
    use_offload: bool,
    use_compression: bool,
    use_spot: bool
):
    """Print cost comparison across providers."""

    print(f"\n{'='*80}")
    print(f"Cost Comparison: {model_size_b}B parameter model, {num_steps:,} steps")
    print(f"Configuration: ZeRO-{zero_stage}, Offload={use_offload}, Compression={use_compression}")
    print(f"Instance Type: {'Spot' if use_spot else 'On-Demand'}")
    print(f"{'='*80}\n")

    # Find suitable instances
    suitable_instances = find_suitable_instances(model_size_b, zero_stage, use_offload)

    if not suitable_instances:
        print("No suitable instances found for this model size!")
        return

    results = []

    for instance in suitable_instances:
        # Estimate training time
        training_hours = estimate_training_time(
            model_size_b=model_size_b,
            num_steps=num_steps,
            num_gpus=instance.num_gpus,
            gpu_type=instance.gpu_type,
            zero_stage=zero_stage,
            use_offload=use_offload,
            use_compression=use_compression
        )

        # Calculate cost
        cost = calculate_cost(instance, training_hours, use_spot)

        results.append({
            "instance": instance,
            "hours": training_hours,
            "cost": cost["total"]
        })

    # Sort by cost
    results.sort(key=lambda x: x["cost"])

    # Print table
    print(f"{'Provider':<12} {'Instance':<25} {'GPU Type':<18} {'GPUs':<6} {'Hours':<8} {'Cost':<12}")
    print(f"{'-'*80}")

    for result in results:
        instance = result["instance"]
        hours = result["hours"]
        cost = result["cost"]

        print(f"{instance.provider.value:<12} {instance.name:<25} {instance.gpu_type:<18} "
              f"{instance.num_gpus:<6} {hours:<8.1f} ${cost:<11,.2f}")

    # Print cheapest
    if results:
        cheapest = results[0]
        print(f"\n{'-'*80}")
        print(f"Cheapest Option: {cheapest['instance'].provider.value} {cheapest['instance'].name}")
        print(f"Total Cost: ${cheapest['cost']:,.2f}")
        print(f"Training Time: {cheapest['hours']:.1f} hours ({cheapest['hours']/24:.1f} days)")
        print(f"{'-'*80}\n")


def calculate_single_config(args):
    """Calculate cost for a single configuration."""

    model_size_b = parse_model_size(args.model_size)

    # Find instance
    provider = Provider(args.provider.lower())
    matching_instances = [i for i in INSTANCES if i.provider == provider and args.gpu_type.upper() in i.gpu_type]

    if not matching_instances:
        print(f"No instances found for provider={args.provider}, gpu_type={args.gpu_type}")
        return

    instance = matching_instances[0]

    # Estimate time
    training_hours = estimate_training_time(
        model_size_b=model_size_b,
        num_steps=args.steps,
        num_gpus=instance.num_gpus * args.num_nodes,
        gpu_type=instance.gpu_type,
        zero_stage=args.zero_stage,
        use_offload=args.offload,
        use_compression=args.compression
    )

    # Calculate cost
    cost = calculate_cost(
        instance=instance,
        training_hours=training_hours * args.num_nodes,  # Cost scales with nodes
        use_spot=args.spot,
        storage_gb=args.storage_gb
    )

    # Print results
    print(f"\n{'='*80}")
    print(f"Training Cost Estimate")
    print(f"{'='*80}\n")

    print(f"Model: {args.model_size} parameters")
    print(f"Training Steps: {args.steps:,}")
    print(f"Provider: {args.provider.upper()}")
    print(f"Instance: {instance.name}")
    print(f"GPUs: {instance.num_gpus} × {args.num_nodes} nodes = {instance.num_gpus * args.num_nodes} total")
    print(f"Instance Type: {'Spot' if args.spot else 'On-Demand'}")
    print(f"\nConfiguration:")
    print(f"  ZeRO Stage: {args.zero_stage}")
    print(f"  CPU Offload: {'Yes' if args.offload else 'No'}")
    print(f"  Compression: {'Yes (1-bit Adam)' if args.compression else 'No'}")

    print(f"\nEstimated Training Time: {training_hours:.1f} hours ({training_hours/24:.1f} days)")

    print(f"\nCost Breakdown:")
    print(f"  Compute: ${cost['compute']:,.2f}")
    print(f"  Storage: ${cost['storage']:,.2f}")
    print(f"  Network: ${cost['network']:,.2f}")
    print(f"  {'-'*40}")
    print(f"  Total: ${cost['total']:,.2f}")

    # Price per GPU per hour
    price_per_gpu = instance.price_per_gpu_hour(args.spot)
    print(f"\nPrice per GPU per hour: ${price_per_gpu:.2f}")

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate DeepSpeed training costs across providers"
    )

    parser.add_argument(
        "--model-size",
        type=str,
        default="7B",
        help="Model size (e.g., 7B, 13B, 70B)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="Number of training steps"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["aws", "gcp", "azure", "lambda", "coreweave"],
        help="Cloud provider"
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        default="A100",
        help="GPU type (e.g., A100, H100, V100)"
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="Number of nodes"
    )
    parser.add_argument(
        "--zero-stage",
        type=int,
        choices=[0, 1, 2, 3],
        default=2,
        help="ZeRO optimization stage"
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Enable CPU offloading"
    )
    parser.add_argument(
        "--compression",
        action="store_true",
        help="Enable gradient compression (1-bit Adam)"
    )
    parser.add_argument(
        "--spot",
        action="store_true",
        help="Use spot/preemptible instances"
    )
    parser.add_argument(
        "--storage-gb",
        type=int,
        default=100,
        help="Storage size in GB"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare costs across all providers"
    )

    args = parser.parse_args()

    if args.compare:
        # Compare all providers
        print_cost_comparison(
            model_size_b=parse_model_size(args.model_size),
            num_steps=args.steps,
            zero_stage=args.zero_stage,
            use_offload=args.offload,
            use_compression=args.compression,
            use_spot=args.spot
        )
    elif args.provider:
        # Calculate for specific configuration
        calculate_single_config(args)
    else:
        print("Please specify --provider or use --compare to compare all providers")
        parser.print_help()


if __name__ == "__main__":
    main()
