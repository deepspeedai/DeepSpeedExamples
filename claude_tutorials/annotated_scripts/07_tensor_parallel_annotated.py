"""
ANNOTATED: Tensor Parallelism with ZeRO-1

Original File: training/tensor_parallel/train.py

This script demonstrates combining Tensor Parallelism with DeepSpeed ZeRO-1:
1. Tensor Parallelism - Split model layers across GPUs
2. ZeRO-1 - Partition optimizer states
3. Simple fine-tuning example on Stanford Alpaca dataset

KEY CONCEPT:
Tensor Parallelism is orthogonal to Data Parallelism (ZeRO).
You can combine them for larger models!

DIFFERENCE FROM MEGATRON-DEEPSPEED:
- This uses transformers library's tensor parallelism (simpler)
- Megatron-LM integration uses Megatron's custom layers (more optimized)
- This example: Good for learning and smaller models
- Megatron: Production-scale training (GPT-3, etc.)

TENSOR PARALLELISM BASICS:
Split a single layer across multiple GPUs.
Example: Linear layer with 4096 input, 4096 output, 2 GPUs
  - GPU 0: Handles first 2048 outputs
  - GPU 1: Handles second 2048 outputs
  - Both compute in parallel, then All-Reduce to combine results
"""

import transformers
from transformers import Trainer, AutoTokenizer
import deepspeed
import torch
import utils  # Custom utilities for data loading


# ============================================================================
# TENSOR PARALLELISM CONFIGURATION
# ============================================================================

"""
TENSOR PARALLELISM SETUP:

In transformers library, tensor parallelism is configured via TrainingArguments:

training_args = TrainingArguments(
    ...
    # [ANNOTATION] Tensor parallelism configuration
    # This tells transformers to split model across GPUs
    deepspeed="ds_config.json",  # DeepSpeed config with tensor parallel settings
    ...
)

DeepSpeed config for tensor parallelism:
{
    "train_batch_size": 32,
    "zero_optimization": {
        "stage": 1  # ZeRO-1: Optimizer state partitioning
    },
    // Tensor parallelism is implicit when using transformers
    // Model will be automatically split if it detects model is too large
}

Note: This example uses ZeRO-1, NOT ZeRO-3!
- ZeRO-1: Partition optimizer states only
- Model parameters are replicated (standard for tensor parallelism)
- Tensor parallelism handles model splitting
"""


# ============================================================================
# DATA LOADING FOR TENSOR PARALLEL + ZERO
# ============================================================================

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> dict:
    """
    [ANNOTATION] **DATA LOADING WITH TENSOR PARALLELISM**

    Important considerations:
    - Each GPU still needs different data (data parallelism)
    - Tensor parallel GPUs process SAME data
    - DeepSpeed's data loader handles this automatically

    Example with 8 GPUs, tensor_parallel_size=2:
    - 4 data parallel groups (2 GPUs each)
    - Within each group: Same data (tensor parallel)
    - Across groups: Different data (data parallel)

    Layout:
      Group 0: GPU 0, GPU 1 → Same data (tensor parallel)
      Group 1: GPU 2, GPU 3 → Different data
      Group 2: GPU 4, GPU 5 → Different data
      Group 3: GPU 6, GPU 7 → Different data
    """
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )


# ============================================================================
# TRAINING WITH TENSOR PARALLELISM + ZERO-1
# ============================================================================

def train():
    """
    [ANNOTATION] Main training function demonstrating tensor parallelism.
    """
    # Parse arguments
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # [ANNOTATION] **LOAD MODEL**
    # Model loaded normally - tensor parallelism applied later
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Set special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = DEFAULT_PAD_TOKEN

    # Resize embeddings if needed
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(),
        tokenizer=tokenizer,
        model=model,
    )

    # Prepare dataset
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # [ANNOTATION] **TRAINER WITH DEEPSPEED**
    # The Trainer class handles:
    # 1. DeepSpeed initialization (including tensor parallelism)
    # 2. Distributed training setup
    # 3. Training loop
    # 4. Checkpointing
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,  # Contains deepspeed config path
        **data_module
    )

    # [ANNOTATION] What happens inside Trainer with DeepSpeed:
    #
    # 1. INITIALIZATION:
    #    - Reads deepspeed config from training_args
    #    - Calls deepspeed.initialize()
    #    - Sets up tensor parallel groups (if configured)
    #    - Applies ZeRO-1 optimizer partitioning
    #
    # 2. MODEL WRAPPING:
    #    - Model is wrapped with DeepSpeed engine
    #    - If tensor parallelism detected: Model split across GPUs
    #    - Each GPU gets a shard of the model
    #
    # 3. FORWARD PASS:
    #    - Input broadcast to all tensor parallel GPUs
    #    - Each GPU computes its shard
    #    - All-Reduce to combine outputs
    #
    # 4. BACKWARD PASS:
    #    - Gradients computed for each shard
    #    - All-Reduce to sync gradients
    #    - ZeRO-1: Optimizer states partitioned across data parallel group
    #
    # 5. OPTIMIZER STEP:
    #    - Each GPU updates its optimizer state partition (ZeRO-1)
    #    - Parameters synchronized across tensor parallel group

    # Train!
    trainer.train()

    # Save model
    trainer.save_model(output_dir=training_args.output_dir)


# ============================================================================
# TENSOR PARALLELISM VS DATA PARALLELISM
# ============================================================================

"""
COMPARISON:
-----------

DATA PARALLELISM (Standard):
  - Replicate full model on each GPU
  - Each GPU processes different data
  - Synchronize gradients after backward
  - Memory: Full model per GPU

  Example (4 GPUs, 8B model):
    GPU 0: [Full 8B model] [Data batch 0]
    GPU 1: [Full 8B model] [Data batch 1]
    GPU 2: [Full 8B model] [Data batch 2]
    GPU 3: [Full 8B model] [Data batch 3]

TENSOR PARALLELISM:
  - Split model across GPUs
  - Each GPU processes SAME data
  - Synchronize intermediate results (All-Reduce)
  - Memory: Model_size / num_gpus per GPU

  Example (4 GPUs, 8B model):
    GPU 0: [2B model shard] [Same data]
    GPU 1: [2B model shard] [Same data]
    GPU 2: [2B model shard] [Same data]
    GPU 3: [2B model shard] [Same data]

COMBINED (Tensor + Data):
  - Split model across tensor parallel group
  - Replicate across data parallel groups
  - Best of both worlds!

  Example (8 GPUs, 8B model, TP=2, DP=4):
    Data Group 0:
      GPU 0: [4B shard] [Data 0]  ← Tensor parallel
      GPU 1: [4B shard] [Data 0]  ← with GPU 0
    Data Group 1:
      GPU 2: [4B shard] [Data 1]
      GPU 3: [4B shard] [Data 1]
    Data Group 2:
      GPU 4: [4B shard] [Data 2]
      GPU 5: [4B shard] [Data 2]
    Data Group 3:
      GPU 6: [4B shard] [Data 3]
      GPU 7: [4B shard] [Data 3]
"""


# ============================================================================
# TENSOR PARALLELISM COMMUNICATION
# ============================================================================

"""
COMMUNICATION PATTERN FOR ONE LAYER:
-------------------------------------

Linear Layer Example (2 GPUs):
  Input: [batch_size, seq_len, hidden_dim]
  Weight: [hidden_dim, output_dim]

Split weight column-wise:
  GPU 0: Weight[:, :output_dim/2]
  GPU 1: Weight[:, output_dim/2:]

Forward:
  1. Broadcast input to both GPUs (if not already there)
  2. GPU 0 computes: output_0 = input @ weight_0
  3. GPU 1 computes: output_1 = input @ weight_1
  4. All-Reduce: Combine output_0 and output_1
  5. Result: Full output on both GPUs

Backward:
  1. Gradient from next layer broadcasted
  2. Each GPU computes gradient for its weight shard
  3. All-Reduce gradient with respect to input
  4. Each GPU has full input gradient for previous layer

Communication Cost:
  - Forward: 1 All-Reduce (size = batch × seq × hidden)
  - Backward: 1 All-Reduce (size = batch × seq × hidden)
  - Total: 2 All-Reduces per layer

For Transformer (attention + FFN):
  - 4 All-Reduces per layer (2 for attention, 2 for FFN)
  - With 32 layers: 128 All-Reduces per forward-backward pass
"""


# ============================================================================
# WHEN TO USE TENSOR PARALLELISM
# ============================================================================

"""
USE TENSOR PARALLELISM WHEN:
✅ Model doesn't fit in single GPU memory
✅ Have fast GPU interconnect (NVLink within node)
✅ Model is too large for data parallelism alone
✅ Training on single node (NVLink available)
✅ Want to increase model capacity without changing architecture

DON'T USE TENSOR PARALLELISM WHEN:
❌ Model fits comfortably in single GPU
❌ Only have slow interconnect (PCIe)
❌ Training across multiple nodes (use pipeline parallelism instead)
❌ Need maximum throughput (communication overhead)

OPTIMAL CONFIGURATION:
- Tensor Parallel: Within node (8 GPUs max, use NVLink)
- Pipeline Parallel: Across nodes
- Data Parallel: Across replicas (with ZeRO)

Example for 64 GPUs (8 nodes):
  - Tensor Parallel Size: 8 (within each node)
  - Pipeline Parallel Size: 4 (across nodes)
  - Data Parallel Size: 2 (2 model replicas)
  - Total: 8 × 4 × 2 = 64 GPUs
"""


# ============================================================================
# ZERO-1 WITH TENSOR PARALLELISM
# ============================================================================

"""
WHY ZERO-1 (NOT ZERO-2 OR ZERO-3)?
-----------------------------------

ZeRO-1: Partition optimizer states only
  - Compatible with tensor parallelism
  - Model parameters NOT partitioned (tensor parallel already splits model)
  - Gradients NOT partitioned (tensor parallel needs full gradients)
  - Only optimizer states partitioned across data parallel group

ZeRO-2/ZeRO-3: Would conflict with tensor parallelism
  - ZeRO-2 partitions gradients → Conflicts with tensor parallel All-Reduce
  - ZeRO-3 partitions parameters → Conflicts with tensor parallel model split
  - Not recommended with tensor parallelism

MEMORY BREAKDOWN (8B model, 8 GPUs, TP=2, DP=4):

Per GPU:
  Model parameters: 8B / 2 (TP) = 4B params × 2 bytes = 8GB
  Gradients: 8B / 2 = 4B grads × 2 bytes = 8GB
  Optimizer states: 8B × 12 bytes / 4 (DP, ZeRO-1) = 24GB
  Activations: ~20GB (depends on batch size)
  Total: ~60GB per GPU

Without ZeRO-1:
  Optimizer states: 8B × 12 bytes / 2 (TP) = 48GB
  Total: ~84GB per GPU → Doesn't fit in 80GB A100!

With ZeRO-1:
  Optimizer states: 24GB (partitioned across DP=4)
  Total: ~60GB → Fits!
"""


# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

"""
1. TENSOR PARALLELISM:
   - Splits individual layers across GPUs
   - Best for within-node (NVLink)
   - Communication: All-Reduce per layer
   - Memory: Model_size / TP_size

2. COMBINE WITH ZERO-1:
   - ZeRO-1 partitions optimizer states
   - Compatible with tensor parallelism
   - Don't use ZeRO-2/3 with tensor parallelism

3. TRANSFORMER INTEGRATION:
   - Use Trainer class with deepspeed config
   - Automatic tensor parallel setup
   - Simpler than Megatron integration

4. COMMUNICATION:
   - 2 All-Reduces per linear layer (forward + backward)
   - 4 All-Reduces per transformer layer (attention + FFN)
   - Requires fast interconnect (NVLink)

5. TYPICAL CONFIGURATION:
   - Tensor Parallel: 2-8 GPUs (within node)
   - Data Parallel: Across nodes
   - ZeRO-1: Partition optimizer states

6. SCALING:
   - TP=2: 2× model capacity
   - TP=4: 4× model capacity
   - TP=8: 8× model capacity (max for single node)
"""


# ============================================================================
# CONFIGURATION EXAMPLE
# ============================================================================

"""
# DeepSpeed config for tensor parallelism + ZeRO-1:
# (ds_config.json)

{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 1,

  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-5,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },

  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16
  },

  "zero_optimization": {
    "stage": 1,  # ZeRO-1 only!

    # Note: No need to specify tensor parallelism here
    # It's handled by transformers library automatically
    # based on model size and available GPUs
  },

  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "wall_clock_breakdown": false
}
"""


# ============================================================================
# USAGE
# ============================================================================

"""
# Launch with tensor parallelism (8 GPUs, TP=2, DP=4):

deepspeed --num_gpus=8 train.py \\
    --model_name_or_path facebook/opt-6.7b \\
    --data_path alpaca_data.json \\
    --output_dir ./output \\
    --num_train_epochs 3 \\
    --per_device_train_batch_size 4 \\
    --per_device_eval_batch_size 4 \\
    --gradient_accumulation_steps 1 \\
    --evaluation_strategy "no" \\
    --save_strategy "steps" \\
    --save_steps 2000 \\
    --save_total_limit 1 \\
    --learning_rate 2e-5 \\
    --weight_decay 0. \\
    --warmup_ratio 0.03 \\
    --lr_scheduler_type "cosine" \\
    --logging_steps 1 \\
    --model_max_length 512 \\
    --deepspeed ./configs/ds_config.json

# The script automatically:
# 1. Detects model is large (6.7B)
# 2. Splits across 2 GPUs (tensor parallel)
# 3. Creates 4 data parallel groups
# 4. Applies ZeRO-1 to partition optimizer states
"""

# [ANNOTATION] See training/tensor_parallel/train.py for full implementation
