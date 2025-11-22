#!/usr/bin/env python3
"""
DeepSpeed Configuration Generator

Interactive CLI tool to generate optimized DeepSpeed configurations
based on your model size, hardware, and training requirements.

Usage:
    python config_generator.py

Or with command-line arguments:
    python config_generator.py --model-size 7B --num-gpus 8 --goal memory

Author: DeepSpeed Community
License: Apache 2.0
"""

import argparse
import json
import sys
from typing import Dict, Any, Optional


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.CYAN}{text}{Colors.ENDC}")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def get_input(prompt: str, default: Optional[str] = None, choices: Optional[list] = None) -> str:
    """Get user input with optional default and validation."""
    if default:
        prompt_text = f"{Colors.BLUE}{prompt} [{default}]: {Colors.ENDC}"
    else:
        prompt_text = f"{Colors.BLUE}{prompt}: {Colors.ENDC}"

    if choices:
        print(f"{Colors.CYAN}Choices: {', '.join(choices)}{Colors.ENDC}")

    while True:
        response = input(prompt_text).strip()
        if not response and default:
            return default

        if choices and response not in choices:
            print_error(f"Invalid choice. Please choose from: {', '.join(choices)}")
            continue

        if response or not default:
            return response


def parse_model_size(size_str: str) -> float:
    """Parse model size string to billions of parameters."""
    size_str = size_str.upper().replace(" ", "")

    if 'B' in size_str:
        return float(size_str.replace('B', ''))
    elif 'M' in size_str:
        return float(size_str.replace('M', '')) / 1000
    else:
        try:
            return float(size_str) / 1e9  # Assume raw param count
        except ValueError:
            print_error(f"Invalid model size: {size_str}")
            sys.exit(1)


def estimate_memory_requirements(model_size_b: float, precision: str = "fp16") -> Dict[str, float]:
    """Estimate memory requirements for model."""
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1
    }

    param_bytes = bytes_per_param.get(precision, 2)
    params = model_size_b * 1e9

    # Memory breakdown (in GB)
    model_memory = (params * param_bytes) / 1e9
    optimizer_memory = model_memory * 2  # Adam states (m, v)
    gradient_memory = model_memory  # Gradients
    activation_memory = model_memory * 0.5  # Rough estimate

    return {
        "model": model_memory,
        "optimizer": optimizer_memory,
        "gradients": gradient_memory,
        "activations": activation_memory,
        "total": model_memory + optimizer_memory + gradient_memory + activation_memory
    }


def choose_zero_stage(model_size_b: float, num_gpus: int, gpu_memory_gb: int, goal: str) -> int:
    """Recommend ZeRO stage based on constraints."""
    memory_req = estimate_memory_requirements(model_size_b)
    total_memory_gb = memory_req["total"]
    memory_per_gpu = total_memory_gb / num_gpus

    if goal == "speed":
        # Prioritize speed
        if memory_per_gpu < gpu_memory_gb * 0.6:
            return 0  # No ZeRO (fastest)
        elif memory_per_gpu < gpu_memory_gb * 0.8:
            return 1  # ZeRO-1
        elif memory_per_gpu < gpu_memory_gb:
            return 2  # ZeRO-2
        else:
            return 3  # ZeRO-3 (memory efficient)

    elif goal == "memory":
        # Prioritize memory efficiency
        if num_gpus >= 8:
            return 3  # ZeRO-3 for maximum efficiency
        elif num_gpus >= 4:
            return 2  # ZeRO-2
        else:
            return 1 # ZeRO-1

    else:  # balanced
        # Balance speed and memory
        if memory_per_gpu < gpu_memory_gb * 0.5:
            return 1  # ZeRO-1 (light optimization)
        elif memory_per_gpu < gpu_memory_gb * 0.9:
            return 2  # ZeRO-2 (balanced)
        else:
            return 3  # ZeRO-3 (necessary for fit)


def generate_config(
    model_size_b: float,
    num_gpus: int,
    gpu_memory_gb: int,
    batch_size: int,
    precision: str,
    goal: str,
    use_offload: bool,
    use_activation_checkpointing: bool
) -> Dict[str, Any]:
    """Generate DeepSpeed configuration."""

    # Choose ZeRO stage
    zero_stage = choose_zero_stage(model_size_b, num_gpus, gpu_memory_gb, goal)

    # Base configuration
    config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        "steps_per_print": 100,
        "wall_clock_breakdown": False
    }

    # Mixed precision
    if precision == "fp16":
        config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        }
    elif precision == "bf16":
        config["bf16"] = {
            "enabled": True
        }

    # ZeRO optimization
    zero_config = {
        "stage": zero_stage
    }

    if zero_stage >= 2:
        zero_config["contiguous_gradients"] = True
        zero_config["overlap_comm"] = True
        zero_config["reduce_bucket_size"] = int(5e8)
        zero_config["allgather_bucket_size"] = int(5e8)

    if zero_stage == 3:
        zero_config["stage3_prefetch_bucket_size"] = "auto"
        zero_config["stage3_param_persistence_threshold"] = "auto"
        zero_config["stage3_max_live_parameters"] = int(1e9)
        zero_config["stage3_max_reuse_distance"] = int(1e9)

    # Offloading
    if use_offload:
        if zero_stage == 3:
            zero_config["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True
            }
            # Only offload params if really needed
            memory_req = estimate_memory_requirements(model_size_b)
            if memory_req["total"] / num_gpus > gpu_memory_gb * 1.2:
                zero_config["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": True
                }
        elif zero_stage == 2:
            zero_config["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True
            }

    config["zero_optimization"] = zero_config

    # Activation checkpointing
    if use_activation_checkpointing:
        config["activation_checkpointing"] = {
            "partition_activations": True,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": True,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        }

    # Optimizer
    config["optimizer"] = {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    }

    # Scheduler
    config["scheduler"] = {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto"
        }
    }

    return config


def print_recommendations(
    model_size_b: float,
    num_gpus: int,
    gpu_memory_gb: int,
    config: Dict[str, Any]
):
    """Print configuration recommendations and analysis."""
    print_header("Configuration Analysis")

    memory_req = estimate_memory_requirements(model_size_b)
    zero_stage = config["zero_optimization"]["stage"]

    # Memory analysis
    print_info("Memory Requirements (without ZeRO):")
    print(f"  Model weights:    {memory_req['model']:>8.2f} GB")
    print(f"  Optimizer states: {memory_req['optimizer']:>8.2f} GB")
    print(f"  Gradients:        {memory_req['gradients']:>8.2f} GB")
    print(f"  Activations:      {memory_req['activations']:>8.2f} GB")
    print(f"  {'─' * 40}")
    print(f"  Total:            {memory_req['total']:>8.2f} GB")
    print(f"  Per GPU ({num_gpus} GPUs):  {memory_req['total']/num_gpus:>8.2f} GB")

    print()

    # ZeRO impact
    if zero_stage == 0:
        print_warning("ZeRO Stage 0: No memory optimization")
        memory_per_gpu = memory_req['total'] / num_gpus
    elif zero_stage == 1:
        memory_per_gpu = (memory_req['model'] + memory_req['gradients'] +
                         memory_req['optimizer'] / num_gpus +
                         memory_req['activations']) / num_gpus
        print_success(f"ZeRO Stage 1: Optimizer state partitioned")
        print(f"  Estimated memory per GPU: {memory_per_gpu:.2f} GB")
    elif zero_stage == 2:
        memory_per_gpu = (memory_req['model'] +
                         (memory_req['optimizer'] + memory_req['gradients']) / num_gpus +
                         memory_req['activations']) / num_gpus
        print_success(f"ZeRO Stage 2: Optimizer + gradient partitioned")
        print(f"  Estimated memory per GPU: {memory_per_gpu:.2f} GB")
    else:  # zero_stage == 3
        memory_per_gpu = ((memory_req['model'] + memory_req['optimizer'] +
                          memory_req['gradients']) / num_gpus +
                         memory_req['activations'])
        print_success(f"ZeRO Stage 3: Full partitioning (model + optimizer + gradients)")
        print(f"  Estimated memory per GPU: {memory_per_gpu:.2f} GB")

    # Fit analysis
    print()
    if memory_per_gpu < gpu_memory_gb * 0.7:
        print_success(f"✓ Model should fit comfortably in {gpu_memory_gb}GB GPU memory")
    elif memory_per_gpu < gpu_memory_gb * 0.9:
        print_warning(f"⚠ Model will fit but memory will be tight ({memory_per_gpu:.1f}GB / {gpu_memory_gb}GB)")
    else:
        print_error(f"✗ Model may not fit in GPU memory ({memory_per_gpu:.1f}GB / {gpu_memory_gb}GB)")
        if "offload_optimizer" in config["zero_optimization"]:
            print_info("  → CPU offloading enabled to help fit model")
        else:
            print_info("  → Consider enabling offloading or using more GPUs")

    # Performance tips
    print()
    print_info("Performance Tips:")
    if zero_stage == 0:
        print("  • Fastest configuration, no communication overhead")
    elif zero_stage == 1:
        print("  • Minimal communication overhead (~5% slower than no ZeRO)")
    elif zero_stage == 2:
        print("  • Moderate communication overhead (~10-15% slower)")
        print("  • overlap_comm enabled to reduce impact")
    else:
        print("  • Higher communication overhead (~20-30% slower)")
        print("  • Prefetching enabled to overlap communication")

    if "activation_checkpointing" in config:
        print("  • Activation checkpointing: 40-60% memory savings, 20-33% slower")

    if "offload_optimizer" in config.get("zero_optimization", {}):
        print("  • CPU offloading: Significant memory savings, 20-40% slower")


def interactive_mode():
    """Run interactive configuration generator."""
    print_header("DeepSpeed Configuration Generator")
    print_info("This tool will help you generate an optimized DeepSpeed configuration")
    print_info("based on your model and hardware specifications.\n")

    # Gather information
    model_size_str = get_input(
        "Model size (e.g., 7B, 13B, 1.5B, 500M)",
        default="7B"
    )
    model_size_b = parse_model_size(model_size_str)

    num_gpus = int(get_input(
        "Number of GPUs",
        default="8"
    ))

    gpu_type = get_input(
        "GPU type",
        default="A100",
        choices=["V100", "A100", "H100", "A6000", "3090", "4090"]
    )

    # GPU memory mapping
    gpu_memory_map = {
        "V100": 32,
        "A100": 80,
        "H100": 80,
        "A6000": 48,
        "3090": 24,
        "4090": 24
    }
    gpu_memory_gb = gpu_memory_map.get(gpu_type, 80)

    batch_size = int(get_input(
        "Desired batch size per GPU",
        default="4"
    ))

    precision = get_input(
        "Precision",
        default="bf16",
        choices=["fp32", "fp16", "bf16", "int8"]
    )

    goal = get_input(
        "Optimization goal",
        default="balanced",
        choices=["speed", "memory", "balanced"]
    )

    use_offload_str = get_input(
        "Enable CPU offloading? (y/n)",
        default="n"
    )
    use_offload = use_offload_str.lower() in ['y', 'yes', 'true', '1']

    use_act_ckpt_str = get_input(
        "Enable activation checkpointing? (y/n)",
        default="n"
    )
    use_activation_checkpointing = use_act_ckpt_str.lower() in ['y', 'yes', 'true', '1']

    # Generate configuration
    print_info("\nGenerating configuration...")
    config = generate_config(
        model_size_b=model_size_b,
        num_gpus=num_gpus,
        gpu_memory_gb=gpu_memory_gb,
        batch_size=batch_size,
        precision=precision,
        goal=goal,
        use_offload=use_offload,
        use_activation_checkpointing=use_activation_checkpointing
    )

    # Print analysis
    print_recommendations(model_size_b, num_gpus, gpu_memory_gb, config)

    # Save configuration
    print()
    output_file = get_input(
        "Output filename",
        default="ds_config.json"
    )

    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)

    print_success(f"Configuration saved to {output_file}")

    # Print usage instructions
    print()
    print_header("Usage Instructions")
    print_info("To use this configuration with DeepSpeed:\n")
    print(f"  {Colors.GREEN}deepspeed --num_gpus={num_gpus} train.py --deepspeed_config={output_file}{Colors.ENDC}\n")
    print_info("Or in your training script:")
    print(f"""
    {Colors.GREEN}import deepspeed

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config_params='{output_file}'
    ){Colors.ENDC}
    """)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate optimized DeepSpeed configurations"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        help="Model size (e.g., 7B, 13B, 500M)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        help="Number of GPUs"
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        choices=["V100", "A100", "H100", "A6000", "3090", "4090"],
        help="GPU type"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16", "int8"],
        help="Training precision"
    )
    parser.add_argument(
        "--goal",
        type=str,
        choices=["speed", "memory", "balanced"],
        help="Optimization goal"
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Enable CPU offloading"
    )
    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help="Enable activation checkpointing"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ds_config.json",
        help="Output filename"
    )

    args = parser.parse_args()

    # If any argument provided, use command-line mode
    if any(vars(args).values()):
        if not all([args.model_size, args.num_gpus, args.gpu_type]):
            print_error("When using command-line mode, --model-size, --num-gpus, and --gpu-type are required")
            sys.exit(1)

        gpu_memory_map = {
            "V100": 32,
            "A100": 80,
            "H100": 80,
            "A6000": 48,
            "3090": 24,
            "4090": 24
        }

        model_size_b = parse_model_size(args.model_size)
        gpu_memory_gb = gpu_memory_map[args.gpu_type]

        config = generate_config(
            model_size_b=model_size_b,
            num_gpus=args.num_gpus,
            gpu_memory_gb=gpu_memory_gb,
            batch_size=args.batch_size or 4,
            precision=args.precision or "bf16",
            goal=args.goal or "balanced",
            use_offload=args.offload,
            use_activation_checkpointing=args.activation_checkpointing
        )

        print_recommendations(model_size_b, args.num_gpus, gpu_memory_gb, config)

        with open(args.output, 'w') as f:
            json.dump(config, f, indent=2)

        print_success(f"\nConfiguration saved to {args.output}")
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
