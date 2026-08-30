"""
ANNOTATED: SuperOffload - ZeRO-3 LLM Fine-Tuning with CPU Offloading

Original File: training/DeepSpeed-SuperOffload/finetune_zero3.py

This script demonstrates advanced DeepSpeed features for training large language models:
1. ZeRO Stage 3 - Full parameter partitioning across GPUs
2. CPU Optimizer (DeepSpeedCPUAdam) - Offload optimizer to CPU
3. Activation Checkpointing/Gradient Checkpointing - Trade computation for memory
4. Flash Attention 2 - Memory-efficient attention implementation
5. Mixed precision (BF16)

MEMORY OPTIMIZATION HIERARCHY:
1. ZeRO-3: Partition parameters across GPUs (each GPU stores 1/N of model)
2. CPU Offload: Move optimizer states to CPU RAM
3. Gradient Checkpointing: Recompute activations instead of storing them
4. Flash Attention 2: Fused, memory-efficient attention kernels

CRITICAL FOR UNDERSTANDING ZeRO-3:
- Parameters are partitioned and only gathered when needed
- During forward: All-Gather parameters, compute, release parameters
- During backward: All-Gather parameters, compute gradients, release parameters
- Only owns 1/N of the model at any time (N = number of GPUs)
"""

import argparse
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepspeed import comm as dist  # [ANNOTATION] DeepSpeed's distributed communication wrapper
from deepspeed.ops.adam import DeepSpeedCPUAdam  # [ANNOTATION] CPU-based Adam optimizer


# ============================================================================
# STEP 1: MODEL LOADING AND PREPARATION
# ============================================================================

def load_model(model_name: str, attn_implementation: str, logger) -> AutoModelForCausalLM:
    """
    [ANNOTATION] Load HuggingFace model with specific configurations.
    """
    logger.debug(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # [ANNOTATION] Load in BF16 to save memory
        attn_implementation=attn_implementation  # flash_attention_2, sdpa, or eager
    )

    return model


def setup_model_training(model: torch.nn.Module,
                         use_activation_checkpointing: bool = True,
                         logger = None) -> None:
    """
    [ANNOTATION] **ACTIVATION CHECKPOINTING** (Gradient Checkpointing)

    This is a critical memory optimization technique:
    - Normally, activations are stored during forward pass for use in backward pass
    - With checkpointing: Discard activations, recompute them during backward
    - Trade-off: Saves memory (~30-40%) at cost of ~30% slower training

    WHEN TO USE:
    - Training very large models that don't fit in GPU memory
    - Increase effective batch size
    - Always use with ZeRO-3 for maximum memory savings
    """
    if use_activation_checkpointing:
        if logger:
            logger.debug("Enabling gradient checkpointing...")

        # [ANNOTATION] Disable KV cache (used for inference, not needed for training)
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False

        # [ANNOTATION] Enable gradient checkpointing
        # use_reentrant=False is recommended for modern PyTorch versions
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )


# ============================================================================
# STEP 2: CPU OPTIMIZER CREATION
# ============================================================================

def create_optimizer(model: AutoModelForCausalLM):
    """
    [ANNOTATION] **CPU OPTIMIZER** - Critical for ZeRO-3 + Offloading

    DeepSpeedCPUAdam:
    - Adam optimizer that runs on CPU instead of GPU
    - Stores optimizer states (momentum, variance) in CPU RAM
    - Only parameter updates happen on CPU, then copied to GPU

    MEMORY FLOW:
    1. Gradients computed on GPU
    2. Gradients copied to CPU
    3. Optimizer update happens on CPU (using CPU RAM for states)
    4. Updated parameters copied back to GPU

    BENEFITS:
    - Offload 12 bytes per parameter (fp32 param + 2 x fp32 optimizer states)
    - For a 7B model: ~84GB of optimizer states moved to CPU
    - Enables training models much larger than GPU memory
    """
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    optimizer = DeepSpeedCPUAdam(
        model.parameters(),
        lr=DEFAULT_OPTIMIZER_LR,
        betas=DEFAULT_OPTIMIZER_BETAS
    )
    return optimizer

    # [ANNOTATION] Alternatives:
    # - FusedAdam: GPU-based, fastest but uses GPU memory
    # - DeepSpeedCPUAdam: CPU-based, slower but saves GPU memory
    # - ZeRO-Offload automatically handles optimizer offload with config


# ============================================================================
# STEP 3: DEEPSPEED INITIALIZATION WITH ZeRO-3
# ============================================================================

def main(args):
    # Load model and tokenizer
    tokenizer = load_tokenizer(args.model_name, logger)
    model = load_model(args.model_name, args.attn_implementation, logger)

    # [ANNOTATION] **CRITICAL FOR MOE MODELS**
    # For Mixture of Experts models, set leaf modules to avoid partitioning experts
    if args.leaf_module:
        from deepspeed.utils import set_z3_leaf_modules
        logger.debug(f"Setting leaf_module to: {args.leaf_module}")
        set_z3_leaf_modules(model, [args.leaf_module])

        # [ANNOTATION] Leaf modules explained:
        # - ZeRO-3 partitions parameters at module granularity
        # - For MoE: Each expert should stay as a single unit (not partitioned)
        # - set_z3_leaf_modules() tells ZeRO-3 to treat these as atomic units

    # Enable activation checkpointing
    setup_model_training(model, args.activation_checkpointing, logger)

    # Create CPU optimizer
    optimizer = create_optimizer(model)

    # Load and preprocess dataset
    tokenized_dataset, train_dataloader = load_and_preprocess_dataset(
        args.dataset_name, args.dataset_percentage, tokenizer, args.max_length, logger
    )

    # [ANNOTATION] **DEEPSPEED INITIALIZATION WITH ZeRO-3**
    # The config is passed via --deepspeed_config argument (JSON file)
    model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,  # Pass CPU optimizer
        training_data=tokenized_dataset,
        collate_fn=default_data_collator
    )

    # [ANNOTATION] What happens during initialize() with ZeRO-3:
    #
    # 1. PARAMETER PARTITIONING:
    #    - Model parameters are partitioned across all GPUs
    #    - Each GPU only stores 1/N of the parameters
    #    - Parameters are converted to "partitioned parameters"
    #
    # 2. OPTIMIZER STATE INITIALIZATION:
    #    - If CPU optimizer: States are created on CPU
    #    - Otherwise: States are partitioned on GPU (1/N per GPU)
    #
    # 3. COMMUNICATION SETUP:
    #    - Sets up All-Gather collectives for forward/backward
    #    - Sets up Reduce-Scatter for gradient synchronization
    #
    # 4. HOOKS INSTALLATION:
    #    - Pre-forward hook: All-Gather parameters before layer
    #    - Post-forward hook: Release parameters after layer
    #    - Pre-backward hook: All-Gather parameters for gradient computation
    #    - Post-backward hook: Reduce-Scatter gradients, release parameters


    # ============================================================================
    # STEP 4: TRAINING LOOP WITH ZERO-3
    # ============================================================================

    model_engine.train()

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}

            # [ANNOTATION] **FORWARD PASS WITH ZERO-3**
            outputs = model_engine(**batch)
            loss = outputs.loss

            # [ANNOTATION] What happens during forward pass:
            #
            # For each transformer layer:
            # 1. PRE-FORWARD HOOK TRIGGERED:
            #    - All-Gather: Collect full parameters from all GPUs
            #    - Example: GPU 0 has params [0-1000], GPU 1 has [1001-2000]
            #              All-Gather brings full [0-2000] to both GPUs
            #
            # 2. LAYER COMPUTATION:
            #    - Standard forward pass with full parameters
            #    - Compute activations
            #    - If gradient checkpointing: Discard activations
            #
            # 3. POST-FORWARD HOOK TRIGGERED:
            #    - Release gathered parameters (free memory)
            #    - Only keep the 1/N partition owned by this GPU
            #
            # Result: Only one layer's parameters in memory at a time!

            # [ANNOTATION] **BACKWARD PASS WITH ZERO-3**
            model_engine.backward(loss)

            # [ANNOTATION] What happens during backward pass:
            #
            # For each transformer layer (in reverse order):
            # 1. PRE-BACKWARD HOOK TRIGGERED:
            #    - If gradient checkpointing: Recompute forward pass for this layer
            #    - All-Gather: Collect full parameters again
            #
            # 2. GRADIENT COMPUTATION:
            #    - Compute gradients with respect to full parameters
            #    - Each GPU computes full gradients
            #
            # 3. POST-BACKWARD HOOK TRIGGERED:
            #    - Reduce-Scatter: Sum gradients across GPUs and partition
            #    - Each GPU gets 1/N of summed gradients (matching param partition)
            #    - Release gathered parameters
            #    - If CPU optimizer: Copy gradients to CPU
            #
            # Result: Each GPU has gradients only for its 1/N parameter partition

            # [ANNOTATION] **OPTIMIZER STEP WITH CPU OFFLOAD**
            model_engine.step()

            # [ANNOTATION] What happens during optimizer step:
            #
            # 1. GRADIENT PROCESSING (on GPU or CPU):
            #    - Apply gradient clipping (if configured)
            #    - Each GPU has 1/N of gradients
            #
            # 2. OPTIMIZER UPDATE (on CPU if using DeepSpeedCPUAdam):
            #    - Load optimizer states from CPU RAM
            #    - Compute Adam update: p = p - lr * m / (sqrt(v) + eps)
            #    - Update momentum (m) and variance (v) states
            #    - Store updated states back to CPU RAM
            #
            # 3. PARAMETER UPDATE:
            #    - If CPU optimizer: Copy updated 1/N parameters from CPU to GPU
            #    - Each GPU now has updated 1/N of parameters
            #
            # 4. CLEANUP:
            #    - Zero gradients
            #    - Increment step counter

            # Note: No explicit all-reduce needed - each GPU updates its partition


# ============================================================================
# MEMORY BREAKDOWN: ZeRO-3 + CPU OFFLOAD
# ============================================================================

"""
EXAMPLE: 7B parameter model, 8 GPUs, BF16 training

WITHOUT ZERO-3:
- Model parameters: 7B × 2 bytes (BF16) = 14GB per GPU
- Gradients: 7B × 2 bytes = 14GB per GPU
- Optimizer states: 7B × 12 bytes (fp32 param + 2×fp32 states) = 84GB per GPU
- Activations: ~40GB per GPU (depends on batch size, sequence length)
- TOTAL: ~152GB per GPU (doesn't fit in 80GB A100!)

WITH ZERO-3 (no offload):
- Model parameters: 7B × 2 bytes / 8 GPUs = 1.75GB per GPU
- Gradients: 7B × 2 bytes / 8 GPUs = 1.75GB per GPU
- Optimizer states: 7B × 12 bytes / 8 GPUs = 10.5GB per GPU
- Activations: ~40GB per GPU
- TOTAL: ~54GB per GPU (fits in 80GB A100)

WITH ZERO-3 + CPU OFFLOAD:
- Model parameters: 7B × 2 bytes / 8 GPUs = 1.75GB per GPU
- Gradients: 7B × 2 bytes / 8 GPUs = 1.75GB per GPU
- Optimizer states: 0GB per GPU (on CPU: 84GB / 8 = 10.5GB CPU RAM per process)
- Activations: ~40GB per GPU
- TOTAL GPU: ~43.5GB per GPU
- TOTAL CPU: ~10.5GB RAM per process

WITH ZERO-3 + CPU OFFLOAD + ACTIVATION CHECKPOINTING:
- Model parameters: 1.75GB per GPU
- Gradients: 1.75GB per GPU
- Optimizer states: 0GB GPU (10.5GB CPU RAM)
- Activations: ~24GB per GPU (40% reduction)
- TOTAL GPU: ~27.5GB per GPU
- TOTAL CPU: ~10.5GB RAM per process
- Can fit 2-3x larger models or 2-3x larger batch sizes!
"""


# ============================================================================
# CONFIGURATION FILE FOR THIS EXAMPLE
# ============================================================================

"""
# ds_config.json for SuperOffload (ZeRO-3 + CPU Offload)

{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",

  "bf16": {
    "enabled": true
  },

  "zero_optimization": {
    "stage": 3,

    // CPU Offloading Configuration
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true  // Faster CPU<->GPU transfers
    },

    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },

    // Communication Optimization
    "overlap_comm": true,  // Overlap AllGather with computation
    "contiguous_gradients": true,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",

    // Memory Management
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,

    // Advanced: Sub-group for very large models
    "sub_group_size": 1e9
  },

  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "wall_clock_breakdown": false
}

CONFIGURATION EXPLAINED:

1. stage: 3
   - Full ZeRO-3 parameter partitioning

2. offload_optimizer.device: "cpu"
   - Move optimizer states to CPU RAM
   - Requires DeepSpeedCPUAdam

3. offload_param.device: "cpu" (optional, for extreme cases)
   - Move parameters to CPU when not in use
   - Slower but enables even larger models

4. pin_memory: true
   - Use pinned (page-locked) CPU memory
   - Faster GPU↔CPU transfers via DMA

5. overlap_comm: true
   - Critical for performance
   - Prefetch next layer's parameters while computing current layer

6. stage3_max_live_parameters
   - Maximum number of full parameters to keep on GPU simultaneously
   - Lower = more memory savings, higher = less communication
"""


# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

"""
1. ZERO-3 PARAMETER LIFECYCLE:
   - Parameters are partitioned (1/N per GPU)
   - Gathered (All-Gather) when needed for computation
   - Released immediately after use
   - Only one layer's full parameters in memory at a time

2. CPU OFFLOADING:
   - Use DeepSpeedCPUAdam for CPU optimizer
   - Optimizer states live in CPU RAM
   - Gradients copied to CPU, updates happen there
   - ~12 bytes per parameter saved on GPU

3. ACTIVATION CHECKPOINTING:
   - Must use with ZeRO-3 for large models
   - Set use_reentrant=False for modern PyTorch
   - Disable KV cache (model.config.use_cache = False)
   - ~30-40% memory reduction, ~30% slower training

4. COMMUNICATION PATTERNS:
   - Forward: All-Gather parameters → Compute → Release
   - Backward: All-Gather parameters → Compute gradients → Reduce-Scatter → Release
   - No manual collective communication needed

5. WHEN TO USE:
   - Model doesn't fit in GPU memory even with smaller batch size
   - Want to train very large models (>7B parameters)
   - Have sufficient CPU RAM (at least 2x model size)
   - Can accept 20-40% slowdown for larger model/batch size
"""

# [ANNOTATION] See training/DeepSpeed-SuperOffload/finetune_zero3.py for full implementation
