"""
ANNOTATED: CIFAR-10 Training with DeepSpeed

Original File: training/cifar/cifar10_deepspeed.py

This script demonstrates a minimal DeepSpeed integration with a simple CNN on CIFAR-10.
It's the simplest example to understand the basic DeepSpeed workflow.

KEY FEATURES:
1. Configurable ZeRO stages (0, 1, 2, 3) via command line
2. Mixed precision training (FP16/BF16/FP32)
3. MoE (Mixture of Experts) support
4. In-memory config dictionary (no external JSON file)
5. Minimal codebase for learning

DISTRIBUTED SETUP:
- Uses deepspeed.init_distributed() for explicit initialization
- Sets device using get_accelerator().set_device()
- Demonstrates proper barrier usage for dataset downloading
"""

import argparse
import os
import deepspeed
import torch
import torch.nn as nn
from deepspeed.accelerator import get_accelerator

# ============================================================================
# STEP 1: ARGUMENT PARSING
# ============================================================================

def add_argument():
    """
    [ANNOTATION] Parse command line arguments.
    This function shows how to add DeepSpeed-specific arguments.
    """
    parser = argparse.ArgumentParser(description="CIFAR")

    # Standard training arguments
    parser.add_argument("-e", "--epochs", default=30, type=int)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local rank passed from distributed launcher")

    # Mixed precision configuration
    parser.add_argument("--dtype", default="fp16", type=str,
                        choices=["bf16", "fp16", "fp32"],
                        help="Datatype used for training")

    # [ANNOTATION] **ZERO STAGE SELECTION**
    # This allows selecting ZeRO optimization stage at runtime
    parser.add_argument("--stage", default=0, type=int,
                        choices=[0, 1, 2, 3],
                        help="ZeRO optimization stage")

    # MoE (Mixture of Experts) arguments
    parser.add_argument("--moe", default=False, action="store_true",
                        help="use deepspeed mixture of experts (moe)")

    # [ANNOTATION] **CRITICAL**: Add DeepSpeed config arguments
    # This adds --deepspeed_config and other DeepSpeed-specific flags
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    return args


# ============================================================================
# STEP 2: DEEPSPEED CONFIGURATION DICTIONARY
# ============================================================================

def get_ds_config(args):
    """
    [ANNOTATION] **CRITICAL FUNCTION**: Build DeepSpeed configuration.

    This function constructs the DeepSpeed config dictionary dynamically
    based on command line arguments. This is an alternative to using a
    separate JSON config file.

    CONFIGURATION STRUCTURE:
    - Training hyperparameters (batch size, logging)
    - Optimizer configuration
    - Learning rate scheduler
    - Mixed precision settings (FP16/BF16)
    - ZeRO optimization settings
    """
    ds_config = {
        # Total batch size = train_batch_size
        # Distributed: train_batch_size = micro_batch * num_gpus * grad_accum_steps
        "train_batch_size": 16,

        # How often to print training stats
        "steps_per_print": 2000,

        # [ANNOTATION] OPTIMIZER CONFIGURATION
        # DeepSpeed will create this optimizer internally
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7,
            },
        },

        # [ANNOTATION] LEARNING RATE SCHEDULER
        # DeepSpeed manages the LR schedule automatically
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 0.001,
                "warmup_num_steps": 1000,
            },
        },

        # Gradient clipping to prevent exploding gradients
        "gradient_clipping": 1.0,

        # [ANNOTATION] MIXED PRECISION CONFIGURATION
        # Enable BF16 or FP16 based on args.dtype
        "bf16": {"enabled": args.dtype == "bf16"},
        "fp16": {
            "enabled": args.dtype == "fp16",
            "fp16_master_weights_and_grads": False,
            "loss_scale": 0,              # 0 = dynamic loss scaling
            "loss_scale_window": 500,
            "hysteresis": 2,
            "min_loss_scale": 1,
            "initial_scale_power": 15,
        },

        # [ANNOTATION] **ZERO OPTIMIZATION CONFIGURATION**
        # This is the core of DeepSpeed's memory savings
        "zero_optimization": {
            # ZeRO stage (0, 1, 2, or 3) - set via command line arg
            "stage": args.stage,

            # [ANNOTATION] Communication optimizations for ZeRO-2 and ZeRO-3
            "allgather_partitions": True,     # All-gather full params in forward
            "reduce_scatter": True,            # Reduce-scatter gradients in backward
            "allgather_bucket_size": 50000000, # Batch allgathers for efficiency
            "reduce_bucket_size": 50000000,    # Batch reduce-scatters for efficiency

            # [ANNOTATION] **OVERLAP COMMUNICATION WITH COMPUTATION**
            # This is critical for performance - hides communication latency
            "overlap_comm": True,

            # Keep gradients contiguous in memory for faster communication
            "contiguous_gradients": True,

            # CPU offloading disabled by default
            # Set to True to offload optimizer states to CPU
            "cpu_offload": False,
        },
    }
    return ds_config


# ============================================================================
# STEP 3: DISTRIBUTED INITIALIZATION
# ============================================================================

def main(args):
    # [ANNOTATION] **EXPLICIT DISTRIBUTED INITIALIZATION**
    # Unlike HelloDeepSpeed, this example explicitly initializes distributed backend
    deepspeed.init_distributed()

    # [ANNOTATION] Get local rank from environment and set device
    _local_rank = int(os.environ.get("LOCAL_RANK"))
    get_accelerator().set_device(_local_rank)

    # [ANNOTATION] **DATASET DOWNLOAD WITH BARRIER**
    # Important pattern: Only rank 0 downloads, others wait
    if torch.distributed.get_rank() != 0:
        # Non-rank-0 processes wait for rank 0 to download data
        torch.distributed.barrier()

    # Load or download CIFAR data (rank 0 does this first)
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    if torch.distributed.get_rank() == 0:
        # Rank 0 signals download is complete
        torch.distributed.barrier()

    # [ANNOTATION] BARRIER PATTERN EXPLANATION:
    # - Prevents race conditions when downloading datasets
    # - Rank 0 downloads first
    # - Other ranks wait at barrier
    # - After rank 0 finishes, it hits the barrier
    # - All ranks proceed to use the downloaded data


    # ============================================================================
    # STEP 4: MODEL DEFINITION
    # ============================================================================

    # [ANNOTATION] Create model (standard PyTorch)
    net = Net(args)

    # Get trainable parameters
    parameters = filter(lambda p: p.requires_grad, net.parameters())

    # [ANNOTATION] For MoE models: Create separate parameter groups for each expert
    # Required when using ZeRO with MoE
    if args.moe_param_group:
        parameters = create_moe_param_groups(net)


    # ============================================================================
    # STEP 5: DEEPSPEED INITIALIZATION
    # ============================================================================

    # [ANNOTATION] Get DeepSpeed config
    ds_config = get_ds_config(args)

    # [ANNOTATION] **CRITICAL**: Initialize DeepSpeed engine
    # This version passes training_data to automatically create a dataloader
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args,                      # Command line arguments
        model=net,                      # PyTorch model
        model_parameters=parameters,    # Model parameters for optimizer
        training_data=trainset,         # Training dataset
        config=ds_config,               # DeepSpeed configuration dict
    )

    # [ANNOTATION] deepspeed.initialize() with training_data:
    # - Automatically creates a DistributedSampler
    # - Creates a DataLoader with the specified batch size
    # - Returns: model_engine, optimizer, dataloader, lr_scheduler

    # Get device information
    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank

    # [ANNOTATION] Determine target dtype for data conversion
    target_dtype = None
    if model_engine.bfloat16_enabled():
        target_dtype = torch.bfloat16
    elif model_engine.fp16_enabled():
        target_dtype = torch.half


    # ============================================================================
    # STEP 6: TRAINING LOOP
    # ============================================================================

    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # Get inputs and labels, move to device
            inputs, labels = data[0].to(local_device), data[1].to(local_device)

            # [ANNOTATION] Convert inputs to target dtype (FP16/BF16)
            if target_dtype is not None:
                inputs = inputs.to(target_dtype)

            # [ANNOTATION] **FORWARD PASS**
            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            # [ANNOTATION] **BACKWARD PASS**
            # DeepSpeed's backward handles gradient partitioning and scaling
            model_engine.backward(loss)

            # [ANNOTATION] **OPTIMIZER STEP**
            # Handles gradient accumulation, optimizer updates, and ZeRO sync
            model_engine.step()

            # [ANNOTATION] What happens in each ZeRO stage:
            #
            # Stage 0 (Disabled):
            #   - Standard data parallelism
            #   - Full model and optimizer on each GPU
            #
            # Stage 1 (Optimizer State Partitioning):
            #   - Each GPU stores 1/N of optimizer states
            #   - Full model and gradients on each GPU
            #   - Memory savings: ~4x for Adam optimizer
            #
            # Stage 2 (Optimizer + Gradient Partitioning):
            #   - Each GPU stores 1/N of optimizer states and gradients
            #   - Full model on each GPU
            #   - Gradients are reduced and partitioned during backward
            #   - Memory savings: ~8x for Adam optimizer
            #
            # Stage 3 (Full Partitioning):
            #   - Each GPU stores 1/N of optimizer, gradients, AND parameters
            #   - Parameters are gathered (All-Gather) during forward/backward
            #   - Memory savings: Can be 64x+ for large models
            #   - Enables training models much larger than GPU memory

            running_loss += loss.item()

            # Logging (only rank 0)
            if local_rank == 0 and i % args.log_interval == (args.log_interval - 1):
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / args.log_interval:.3f}")
                running_loss = 0.0

    print("Finished Training")


# ============================================================================
# KEY TAKEAWAYS FOR CIFAR EXAMPLE
# ============================================================================

"""
1. MINIMAL INTEGRATION:
   - Only ~10 lines of code changed from standard PyTorch
   - Main changes: deepspeed.initialize(), model_engine.backward(), model_engine.step()

2. CONFIGURATION:
   - Can use in-memory dict (this example) or external JSON file
   - ZeRO stage can be selected at runtime via command line

3. DISTRIBUTED BEST PRACTICES:
   - Use barriers when downloading datasets
   - Only rank 0 should do I/O operations when possible
   - Check local_rank before printing/logging

4. MIXED PRECISION:
   - Automatic loss scaling for FP16
   - BF16 doesn't need loss scaling
   - DeepSpeed handles precision conversion

5. MoE SUPPORT:
   - DeepSpeed has built-in MoE layers
   - Requires special parameter grouping for ZeRO optimization
"""


# ============================================================================
# COMMAND LINE USAGE
# ============================================================================

"""
Run with ZeRO Stage 0 (baseline):
    deepspeed --num_gpus=4 cifar10_deepspeed.py --stage 0 --dtype fp16

Run with ZeRO Stage 1:
    deepspeed --num_gpus=4 cifar10_deepspeed.py --stage 1 --dtype fp16

Run with ZeRO Stage 2:
    deepspeed --num_gpus=4 cifar10_deepspeed.py --stage 2 --dtype fp16

Run with ZeRO Stage 3:
    deepspeed --num_gpus=4 cifar10_deepspeed.py --stage 3 --dtype fp16

With BF16 instead of FP16:
    deepspeed --num_gpus=4 cifar10_deepspeed.py --stage 2 --dtype bf16

With MoE:
    deepspeed --num_gpus=4 cifar10_deepspeed.py --stage 2 --moe --moe_param_group
"""

# [ANNOTATION] See training/cifar/cifar10_deepspeed.py for full implementation
