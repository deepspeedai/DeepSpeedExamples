"""
ANNOTATED: ZenFlow - ZeRO-2 with Sparse Optimizer Updates

Original File: training/DeepSpeed-ZenFlow/finetuning/finetune_llama.py

This script demonstrates ZeRO-2 optimization with ZenFlow, a novel technique for:
1. ZeRO Stage 2 - Optimizer state + gradient partitioning
2. ZenFlow - Sparse optimizer state updates with CPU offloading
3. Selective parameter updates to reduce CPU↔GPU communication
4. Overlap optimizer updates with forward pass

ZENFLOW INNOVATION:
- Traditional CPU offload: ALL optimizer states moved to CPU, ALL updated every step
- ZenFlow: Only update TOP-K most important optimizer states each step
- Reduces CPU↔GPU transfer by 90% while maintaining training quality
- Overlaps CPU optimizer updates with GPU forward pass

KEY DIFFERENCE FROM ZERO-3:
- ZeRO-2: Full model on each GPU (no parameter partitioning)
- Only gradients and optimizer states are partitioned
- Simpler than ZeRO-3, good for models that fit in GPU memory
"""

import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepspeed import comm as dist

# ============================================================================
# ZENFLOW CONFIGURATION (Embedded in ds_config.json)
# ============================================================================

"""
# zf_config.json - ZeRO-2 + ZenFlow Configuration

{
    "train_batch_size": 8,
    "bf16": { "enabled": true },

    // [ANNOTATION] **ZERO-2 CONFIGURATION**
    "zero_optimization": {
      "stage": 2,  // Optimizer + Gradient partitioning (NOT parameter partitioning)

      // [ANNOTATION] **CPU OFFLOADING** - Move optimizer states to CPU
      "offload_optimizer": {
        "device": "cpu",
        "pin_memory": true  // Pinned memory for faster GPU↔CPU transfers
      },

      // [ANNOTATION] **ZENFLOW CONFIGURATION** - The innovation!
      "zenflow": {
            // Only update top 10% of optimizer states each step
            "topk_ratio": 0.1,

            // Update interval: Run ZenFlow selection every 4 steps
            "update_interval": 4,

            // Warm-up: Do full updates for first N rounds (0 = no warmup)
            "full_warm_up_rounds": 0,

            // Overlap optimizer step with forward pass
            "overlap_step": true
        }
    },

    "optimizer": {
      "type": "AdamW",
      "params": {
        "lr": 2e-5,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01
      }
    },

    "gradient_accumulation_steps": 1,
    "gradient_clipping": 1.0,
    "zero_allow_untested_optimizer": true
}
"""


# ============================================================================
# ZENFLOW ALGORITHM EXPLANATION
# ============================================================================

"""
TRADITIONAL ZERO-2 + CPU OFFLOAD:
----------------------------------
Every optimization step:
1. All gradients: GPU → CPU  (100% transfer)
2. Update all optimizer states on CPU (100% computation)
3. All parameters: CPU → GPU (100% transfer)

For 7B model with 8 GPUs:
- Each GPU has 1/8 of optimizer states (~10GB on CPU per process)
- Every step: Transfer 10GB to CPU, update, transfer 10GB back
- Bottleneck: CPU↔GPU bandwidth (PCIe ~32GB/s)


ZENFLOW OPTIMIZATION:
---------------------
Insight: Not all parameters need frequent updates!
- Some parameters change rapidly (important)
- Some parameters change slowly (less important)

ZenFlow selects TOP-K most important optimizer states to update:

Every 'update_interval' steps:
1. Compute importance score for each parameter
   - Based on gradient magnitude, update history, etc.
2. Select top-k% most important parameters
3. Mark these for update

Each optimization step:
1. Only selected gradients: GPU → CPU (~10% transfer)
2. Update only selected optimizer states on CPU (~10% computation)
3. Only updated parameters: CPU → GPU (~10% transfer)

Result: 10x reduction in CPU↔GPU transfer, minimal accuracy loss!


OVERLAP OPTIMIZATION:
--------------------
With overlap_step=true:

Traditional (Sequential):
  [Forward Pass] → [Backward Pass] → [Optimizer Step] → [Next Forward]
                                     ↑ CPU update blocks GPU ↑

ZenFlow (Overlapped):
  [Forward Pass] → [Backward Pass] → [Next Forward]
                   ↑
                   [Optimizer Step on CPU (overlapped)]

The CPU optimizer update happens asynchronously while GPU computes next forward!
"""


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main(args):
    # [ANNOTATION] Simple training script - ZenFlow magic is in the config!

    # Set random seed
    set_seed(args.seed)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # [ANNOTATION] Load model in BF16 to save memory
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16
    )

    # Load and tokenize dataset
    dataset = load_dataset("tatsu-lab/alpaca")
    tokenized_dataset = dataset["train"].map(
        lambda x: preprocess_alpaca(x, tokenizer),
        batched=False
    )

    # [ANNOTATION] **DEEPSPEED INITIALIZATION**
    # ZenFlow is configured via JSON file passed with --deepspeed argument
    model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
        args=args,  # Contains --deepspeed path to zf_config.json
        model=model,
        model_parameters=model.parameters(),
        training_data=tokenized_dataset,
        collate_fn=default_data_collator
    )

    # [ANNOTATION] What deepspeed.initialize() does with ZeRO-2 + ZenFlow:
    #
    # 1. GRADIENT PARTITIONING:
    #    - Partition gradients across GPUs (each GPU has 1/N)
    #    - Set up Reduce-Scatter collective for gradient synchronization
    #
    # 2. OPTIMIZER STATE PARTITIONING + CPU OFFLOAD:
    #    - Each GPU's optimizer states moved to CPU (1/N per process)
    #    - Allocate pinned memory for fast transfers
    #
    # 3. ZENFLOW INITIALIZATION:
    #    - Initialize importance scores for all parameters
    #    - Prepare top-k selection buffers
    #    - Set up async optimizer step if overlap_step=true
    #
    # 4. MODEL (NOT PARTITIONED):
    #    - Full model stays on each GPU
    #    - No parameter partitioning (that's ZeRO-3)


    # ============================================================================
    # TRAINING LOOP WITH ZENFLOW
    # ============================================================================

    model_engine.train()
    global_step = 0

    for epoch in range(args.num_train_epochs):
        if dist.get_rank() == 0:
            print(f"Starting epoch {epoch + 1}/{args.num_train_epochs}")

        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}

            # [ANNOTATION] **FORWARD PASS** (Standard with ZeRO-2)
            outputs = model_engine(**batch)
            loss = outputs.loss

            # [ANNOTATION] ZeRO-2 forward pass:
            # - Full model on each GPU
            # - Standard forward computation
            # - Activations stored in GPU memory


            # [ANNOTATION] **BACKWARD PASS** (ZeRO-2 gradient partitioning)
            model_engine.backward(loss)

            # [ANNOTATION] What happens in backward with ZeRO-2:
            #
            # 1. GRADIENT COMPUTATION:
            #    - Compute full gradients on each GPU
            #
            # 2. GRADIENT REDUCE-SCATTER:
            #    - Sum gradients across all GPUs
            #    - Partition summed gradients (each GPU keeps 1/N)
            #    - Example with 4 GPUs:
            #      GPU 0: keeps gradients for params [0-1000]
            #      GPU 1: keeps gradients for params [1001-2000]
            #      GPU 2: keeps gradients for params [2001-3000]
            #      GPU 3: keeps gradients for params [3001-4000]
            #
            # 3. MOVE TO CPU (with offload_optimizer):
            #    - Each GPU's gradient partition moved to CPU
            #
            # 4. ZENFLOW SELECTION (every update_interval steps):
            #    - Compute importance scores
            #    - Select top-k% gradients to actually use for update
            #    - Discard other gradients (no optimizer update needed)


            # [ANNOTATION] **OPTIMIZER STEP** (ZenFlow magic happens here!)
            model_engine.step()

            # [ANNOTATION] What happens in optimizer step with ZenFlow:
            #
            # WITHOUT ZENFLOW (traditional ZeRO-2 + CPU offload):
            # 1. All gradients on CPU (1/N per process)
            # 2. Update all optimizer states on CPU
            # 3. Copy all updated parameters back to GPU
            # 4. All-Gather updated parameters across GPUs
            #
            # WITH ZENFLOW:
            # 1. Only top-k selected gradients on CPU (~10%)
            # 2. Update only selected optimizer states on CPU (~10% computation)
            # 3. Copy only updated parameters back to GPU (~10% transfer)
            # 4. All-Gather only updated parameters
            # 5. If overlap_step=true: Steps 2-4 happen asynchronously!
            #
            # OVERLAP DETAIL:
            # - Optimizer step launched on CPU worker thread
            # - GPU immediately proceeds to next forward pass
            # - When optimizer finishes, parameters are updated
            # - Next backward will use updated parameters

            global_step += 1

            if dist.get_rank() == 0:
                print(f"Step {global_step}, Loss: {loss.item():.4f}")


    # [ANNOTATION] Save model
    if dist.get_rank() == 0:
        model_engine.save_checkpoint(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


# ============================================================================
# PERFORMANCE COMPARISON
# ============================================================================

"""
EXAMPLE: LLaMA-13B on 8x A100-80GB

ZERO-2 (No Offload):
- Memory: ~45GB per GPU (model + optimizer states + gradients)
- Throughput: 100% (baseline)

ZERO-2 + CPU Offload (No ZenFlow):
- Memory: ~30GB per GPU (model + gradients only)
- Throughput: ~70% (30% slowdown due to CPU↔GPU transfer)

ZERO-2 + CPU Offload + ZenFlow (topk_ratio=0.1):
- Memory: ~30GB per GPU (same as above)
- Throughput: ~92% (only 8% slowdown!)
- Accuracy: ~99.5% of full training (minimal degradation)

KEY INSIGHT:
ZenFlow recovers most of the performance lost to CPU offloading
while maintaining memory savings!
"""


# ============================================================================
# ZENFLOW HYPERPARAMETERS
# ============================================================================

"""
1. topk_ratio:
   - What percentage of parameters to update each step
   - Typical: 0.1 (10%) to 0.3 (30%)
   - Lower = more memory/computation savings, might affect convergence
   - Higher = better convergence, less savings

2. update_interval:
   - How often to recompute importance scores
   - Typical: 1 to 10 steps
   - Lower = more accurate selection, more overhead
   - Higher = less overhead, might miss important parameters

3. full_warm_up_rounds:
   - Number of initial rounds to do full updates
   - Typical: 0 to 100 steps
   - Helps stabilize training at the beginning

4. overlap_step:
   - Whether to overlap CPU optimizer with GPU forward
   - Always set to true if possible
   - Requires asynchronous execution support
"""


# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

"""
1. ZERO-2 vs ZERO-3:
   - ZeRO-2: Partition gradients + optimizer states, full model on each GPU
   - ZeRO-3: Partition everything (gradients + optimizer + parameters)
   - ZeRO-2 is simpler, faster, but requires model to fit in GPU memory

2. ZENFLOW INNOVATION:
   - Sparse optimizer updates: Only update important parameters
   - Reduces CPU↔GPU transfer by 90%
   - Overlaps CPU work with GPU computation
   - Minimal accuracy loss (<0.5%)

3. WHEN TO USE ZERO-2:
   - Model fits in GPU memory but optimizer states don't
   - Example: 7B-13B models on A100-40GB/80GB
   - Want simpler setup than ZeRO-3
   - Don't need extreme memory savings

4. WHEN TO ADD ZENFLOW:
   - Using ZeRO-2 with CPU offload
   - CPU↔GPU bandwidth is bottleneck
   - Can accept slight accuracy trade-off for speed

5. GRADIENT FLOW (ZERO-2):
   Forward:  Each GPU has full model → Compute activations
   Backward: Compute gradients → Reduce-Scatter (partition gradients)
   Optimizer: Each GPU updates its 1/N parameters → All-Gather parameters
"""

# [ANNOTATION] See training/DeepSpeed-ZenFlow/finetuning/finetune_llama.py for full implementation
