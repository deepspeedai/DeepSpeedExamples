"""
ANNOTATED: Domino - DeepSpeed + Megatron-LM Integration

Original File: training/DeepSpeed-Domino/pretrain_gpt.py

This script demonstrates the integration of DeepSpeed with Megatron-LM for:
1. Tensor Parallelism (split model across GPUs within a node)
2. Pipeline Parallelism (split layers across nodes)
3. Data Parallelism (via DeepSpeed ZeRO)
4. 3D Parallelism (Tensor + Pipeline + Data)

KEY CONCEPTS:
- Megatron-LM: NVIDIA's framework for model parallelism
- Tensor Parallelism: Split individual layers across GPUs
- DeepSpeed: Handles data parallelism and ZeRO optimizations
- Domino: Efficient combination of Megatron + DeepSpeed

WHEN TO USE:
- Very large models (100B+ parameters)
- Multi-node training
- Need both model parallelism and data parallelism
- Training models larger than what data parallelism alone can handle

ARCHITECTURE:
Model is GPT-3 style transformer with Megatron's tensor parallel layers.
"""

# Copyright (c) 2022, NVIDIA CORPORATION.
# Adapted from Megatron-LM's pretrain_gpt.py

from functools import partial
import torch
from megatron import get_args, get_timers, get_tokenizer, print_rank_0
from megatron.arguments import core_transformer_config_from_args
from megatron.core import tensor_parallel  # [ANNOTATION] Megatron's tensor parallelism
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.utils import get_ltor_masks_and_position_ids, average_losses_across_data_parallel_group

from domino.gpt_model import GPTModel  # [ANNOTATION] Megatron GPT model
from domino.training import pretrain   # [ANNOTATION] Domino's training loop


# ============================================================================
# TENSOR PARALLELISM EXPLANATION
# ============================================================================

"""
TENSOR PARALLELISM:
-------------------
Split individual layers across multiple GPUs (typically within a node).

Example: Transformer layer with 4 GPUs (tensor_parallel_size=4)

Standard (no parallelism):
  GPU 0: [Full Attention] [Full FFN]

Tensor Parallel (split across 4 GPUs):
  GPU 0: [Attn Q,K,V: 1/4] [FFN: 1/4]
  GPU 1: [Attn Q,K,V: 1/4] [FFN: 1/4]
  GPU 2: [Attn Q,K,V: 1/4] [FFN: 1/4]
  GPU 3: [Attn Q,K,V: 1/4] [FFN: 1/4]

Communication:
  - All-Reduce after attention (combine results from all GPUs)
  - All-Reduce after FFN (combine results)

Memory savings: ~4× for model parameters
Communication: 2 All-Reduces per layer
"""


# ============================================================================
# PIPELINE PARALLELISM EXPLANATION
# ============================================================================

"""
PIPELINE PARALLELISM:
---------------------
Split layers across different GPUs/nodes (typically across nodes).

Example: 32-layer model with 4 pipeline stages:

Stage 0 (GPU 0-7):   Layers 0-7
Stage 1 (GPU 8-15):  Layers 8-15
Stage 2 (GPU 16-23): Layers 16-23
Stage 3 (GPU 24-31): Layers 24-31

Forward pass:
  Stage 0 processes micro-batch 1 → passes to Stage 1
  Stage 0 processes micro-batch 2 → passes to Stage 1
  ...
  All stages process different micro-batches in parallel

Backward pass (reverse order):
  Stage 3 → Stage 2 → Stage 1 → Stage 0

Memory savings: ~4× for model parameters
Communication: Activations passed between stages
"""


# ============================================================================
# 3D PARALLELISM
# ============================================================================

"""
3D PARALLELISM: Tensor + Pipeline + Data Parallelism
-----------------------------------------------------

Example: 175B parameter model on 512 GPUs (64 nodes × 8 GPUs/node)

Configuration:
  - Tensor Parallel: 8 (split within node)
  - Pipeline Parallel: 8 (split across nodes)
  - Data Parallel: 8 (ZeRO replication)

Layout:
  64 GPUs form one model replica (8 tensor × 8 pipeline)
  8 model replicas for data parallelism
  Total: 64 × 8 = 512 GPUs

GPU assignment:
  Tensor Parallel Group: GPUs within same node
  Pipeline Parallel Group: Same GPU rank across nodes
  Data Parallel Group: Same position in different replicas

Memory:
  Model size: 175B params
  Per GPU: 175B / (8 tensor × 8 pipeline) = ~2.7B params
  Can fit in 40GB A100!
"""


# ============================================================================
# MODEL BUILDER
# ============================================================================

def model_builder(pre_process=True, post_process=True):
    """
    [ANNOTATION] **MODEL CREATION WITH MEGATRON**

    Args:
        pre_process: Whether this is the first pipeline stage
                     (includes embeddings)
        post_process: Whether this is the last pipeline stage
                      (includes final layer norm and loss)

    Pipeline Parallelism:
    - First stage (pre_process=True): Has embedding layer
    - Middle stages: Only transformer layers
    - Last stage (post_process=True): Has output layer

    Tensor Parallelism:
    - All stages have tensor-parallel layers
    - Handled internally by Megatron
    """
    print_rank_0('Building GPT model ...')

    # [ANNOTATION] Get Megatron config
    config = core_transformer_config_from_args(get_args())

    # [ANNOTATION] Create Megatron GPT model
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,      # Output is kept partitioned (for pipeline)
        pre_process=pre_process,   # First stage has embeddings
        post_process=post_process  # Last stage has output projection
    )

    # [ANNOTATION] What GPTModel does:
    # 1. Creates transformer layers with tensor parallelism
    # 2. Distributes layers across pipeline stages
    # 3. Sets up communication groups (tensor, pipeline, data)
    # 4. Wraps parameters for Megatron's parallel training

    return model


# ============================================================================
# DATASET BUILDER
# ============================================================================

def dataset_builder(train_val_test_num_samples):
    """
    [ANNOTATION] Build datasets for pre-training.

    Important: Each data parallel rank processes different data.
    """
    args = get_args()
    print_rank_0('Load GPT dataset ...')

    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
        data_cache_path=args.data_cache_path
    )

    return train_ds, valid_ds, test_ds


# ============================================================================
# FORWARD STEP (WITH PIPELINE PARALLELISM)
# ============================================================================

def forward_step(data_iterator, model):
    """
    [ANNOTATION] **FORWARD STEP FOR PIPELINE PARALLELISM**

    Critical differences from standard forward:
    - Must handle pipeline communication
    - Only certain stages compute loss
    - Activations passed between stages
    """
    timers = get_timers()

    # Get batch
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers('batch-generator').stop()

    # [ANNOTATION] **FORWARD PASS**
    # For pipeline parallelism:
    # - First stage (pre_process=True): Receives tokens, computes embeddings
    # - Middle stages: Receive activations from previous stage
    # - Last stage (post_process=True): Computes loss
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    # [ANNOTATION] What happens during forward in pipeline:
    #
    # FIRST STAGE (rank 0 in pipeline):
    #   1. Embed tokens
    #   2. Process through its transformer layers
    #   3. Send activations to next stage (send operation)
    #   4. output_tensor = None (no loss computed here)
    #
    # MIDDLE STAGES:
    #   1. Receive activations from previous stage (recv operation)
    #   2. Process through its transformer layers
    #   3. Send activations to next stage (send operation)
    #   4. output_tensor = None
    #
    # LAST STAGE:
    #   1. Receive activations from previous stage
    #   2. Process through its transformer layers
    #   3. Compute loss
    #   4. output_tensor = loss (only last stage has loss!)

    return output_tensor, partial(loss_func, loss_mask)


def get_batch(data_iterator):
    """
    [ANNOTATION] **TENSOR PARALLEL BROADCAST**

    Critical pattern: Broadcast data from tensor-parallel rank 0.

    Why?
    - Only one process in tensor parallel group loads data
    - Broadcast to all others in the group
    - Ensures all GPUs in tensor group have same data
    """
    args = get_args()
    tokenizer = get_tokenizer()

    keys = ['text']
    datatype = torch.int64

    # [ANNOTATION] Only tensor parallel rank 0 loads data
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    # [ANNOTATION] **BROADCAST TO TENSOR PARALLEL GROUP**
    # All GPUs in same tensor parallel group must have same input
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack batch
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get masks and position ids
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss
    )

    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask, output_tensor):
    """
    [ANNOTATION] **LOSS COMPUTATION**

    Important: Only the last pipeline stage computes loss.
    Other stages return None for output_tensor.
    """
    raw_loss = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(raw_loss * loss_mask) / loss_mask.sum()

    # [ANNOTATION] Reduce loss across data parallel group
    # All replicas of the model should have same average loss
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # [ANNOTATION] **PRETRAIN FUNCTION FROM DOMINO**
    # This is Domino's main training loop that:
    # 1. Sets up DeepSpeed engine
    # 2. Initializes Megatron parallelism
    # 3. Runs training with pipeline scheduling
    # 4. Handles checkpointing

    pretrain(
        model_builder,     # Function to create model
        dataset_builder,   # Function to create datasets
        forward_step       # Function for forward pass
    )

    # [ANNOTATION] What pretrain() does internally:
    #
    # 1. INITIALIZE PARALLELISM:
    #    - Set up tensor parallel groups
    #    - Set up pipeline parallel groups
    #    - Set up data parallel groups
    #    - Initialize communication backends
    #
    # 2. CREATE MODEL:
    #    - Call model_builder for each pipeline stage
    #    - Distribute layers across pipeline stages
    #    - Apply tensor parallelism within each stage
    #
    # 3. INITIALIZE DEEPSPEED:
    #    - Wrap model with DeepSpeed engine
    #    - Apply ZeRO optimizations (for data parallelism)
    #    - Set up optimizer with DeepSpeed
    #
    # 4. TRAINING LOOP:
    #    - Pipeline scheduling (interleaved micro-batches)
    #    - Forward passes through pipeline
    #    - Backward passes (reverse order)
    #    - Gradient accumulation
    #    - Optimizer step (synchronized across data parallel group)
    #
    # 5. CHECKPOINTING:
    #    - Save model state (all pipeline stages)
    #    - Save optimizer state (ZeRO partitioned)


# ============================================================================
# COMMUNICATION PATTERNS IN 3D PARALLELISM
# ============================================================================

"""
COMMUNICATION GROUPS:
---------------------

1. TENSOR PARALLEL GROUP:
   - GPUs within same node (typically 8 GPUs)
   - Operations: All-Reduce (after attention, after FFN)
   - Frequency: 2× per transformer layer
   - Bandwidth: NVLink (very fast, 600 GB/s)

2. PIPELINE PARALLEL GROUP:
   - GPUs at same position across nodes
   - Operations: Send/Recv (pass activations between stages)
   - Frequency: Per micro-batch
   - Bandwidth: InfiniBand (fast, 200 Gb/s)

3. DATA PARALLEL GROUP (ZeRO):
   - Same model position across different replicas
   - Operations: Reduce-Scatter (gradients), All-Gather (parameters)
   - Frequency: Once per optimization step
   - Bandwidth: InfiniBand (cross-node)

EXAMPLE WITH 64 GPUs (8 nodes × 8 GPUs/node):
  Tensor Parallel Size: 8
  Pipeline Parallel Size: 8
  Data Parallel Size: 1 (no data parallelism in this example)

GPU Layout:
  Node 0: GPUs 0-7   → Pipeline Stage 0, Tensor Group 0
  Node 1: GPUs 8-15  → Pipeline Stage 1, Tensor Group 0
  Node 2: GPUs 16-23 → Pipeline Stage 2, Tensor Group 0
  ...
  Node 7: GPUs 56-63 → Pipeline Stage 7, Tensor Group 0

Tensor Parallel Groups:
  Group 0: [GPU 0, GPU 1, GPU 2, ..., GPU 7]    (within Node 0)
  Group 1: [GPU 8, GPU 9, GPU 10, ..., GPU 15]  (within Node 1)
  ...

Pipeline Parallel Groups:
  Group 0: [GPU 0, GPU 8, GPU 16, ..., GPU 56]  (rank 0 on each node)
  Group 1: [GPU 1, GPU 9, GPU 17, ..., GPU 57]  (rank 1 on each node)
  ...
"""


# ============================================================================
# PIPELINE SCHEDULING
# ============================================================================

"""
INTERLEAVED PIPELINE SCHEDULING:
---------------------------------

Problem with naive pipeline:
  Forward: Stage0 → Stage1 → Stage2 → Stage3
           Stage0 idle while Stage1-3 work!

Solution: Interleaved micro-batches

Timeline (4 micro-batches, 4 stages):

Time  | Stage 0      | Stage 1      | Stage 2      | Stage 3
------|--------------|--------------|--------------|-------------
T0    | F0           | -            | -            | -
T1    | F1           | F0           | -            | -
T2    | F2           | F1           | F0           | -
T3    | F3           | F2           | F1           | F0
T4    | B0           | F3           | F2           | F1
T5    | B1           | B0           | F3           | F2
T6    | B2           | B1           | B0           | F3
T7    | B3           | B2           | B1           | B0
T8    | -            | B3           | B2           | B1
T9    | -            | -            | B3           | B2
T10   | -            | -            | -            | B3

F = Forward, B = Backward

Pipeline efficiency: ~75% (vs ~25% naive)
"""


# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

"""
1. MEGATRON INTEGRATION:
   - Tensor parallelism for within-node model splitting
   - Pipeline parallelism for across-node model splitting
   - DeepSpeed handles data parallelism (ZeRO)

2. 3D PARALLELISM:
   - Tensor: Split layers across GPUs (NVLink)
   - Pipeline: Split layers across nodes (InfiniBand)
   - Data: Replicate model across groups (ZeRO)

3. COMMUNICATION GROUPS:
   - Tensor: All-Reduce within node (fast)
   - Pipeline: Send/Recv between nodes (medium)
   - Data: ZeRO operations across replicas (slower)

4. FORWARD STEP:
   - Pipeline stages process sequentially
   - Tensor parallel groups process in parallel
   - Only last stage computes loss

5. BROADCAST PATTERN:
   - Tensor parallel rank 0 loads data
   - Broadcast to all ranks in tensor group
   - Ensures consistent inputs

6. WHEN TO USE:
   - Model > 100B parameters
   - Multi-node training required
   - Need maximum memory efficiency
   - Have fast interconnect (NVLink + InfiniBand)

7. DOMINO CONTRIBUTION:
   - Efficient integration of Megatron + DeepSpeed
   - Optimized pipeline scheduling
   - Minimal communication overhead
   - Achieved record training speeds for GPT-3
"""


# ============================================================================
# CONFIGURATION EXAMPLE
# ============================================================================

"""
# Launch script for GPT-3 13B with 3D parallelism:

deepspeed --num_nodes 8 --num_gpus 8 pretrain_gpt.py \\
    --tensor-model-parallel-size 2 \\    # Split within node
    --pipeline-model-parallel-size 4 \\  # Split across nodes
    --num-layers 40 \\                   # 40 transformer layers
    --hidden-size 5120 \\                # Hidden dimension
    --num-attention-heads 40 \\          # Attention heads
    --micro-batch-size 1 \\              # Per pipeline stage
    --global-batch-size 128 \\           # Total batch size
    --seq-length 2048 \\                 # Sequence length
    --max-position-embeddings 2048 \\
    --train-iters 500000 \\
    --lr 1.5e-4 \\
    --lr-decay-style cosine \\
    --min-lr 1.5e-5 \\
    --weight-decay 0.1 \\
    --clip-grad 1.0 \\
    --warmup 0.01 \\
    --fp16 \\                            # Mixed precision
    --zero-stage 1 \\                    # DeepSpeed ZeRO-1
    --data-path /data/gpt/corpus

# Calculation:
# - 64 GPUs total (8 nodes × 8 GPUs)
# - Tensor: 2 GPUs per layer
# - Pipeline: 4 stages × 10 layers each
# - Data parallel: 64 / (2 × 4) = 8 replicas
# - Model replicated 8 times with ZeRO-1
"""

# [ANNOTATION] See training/DeepSpeed-Domino/pretrain_gpt.py for full implementation
