"""
ANNOTATED: Bing BERT - Production-Scale BERT Pre-training

Original File: training/bing_bert/deepspeed_train.py

This script demonstrates Microsoft's production BERT training that achieved:
- Fastest BERT training record: 44 minutes on 1024 V100 GPUs
- Full BERT-Large pre-training on Wikipedia + BookCorpus
- Production-scale distributed training patterns

KEY PRODUCTION PATTERNS:
1. Custom dataset provider (not simple DataLoader)
2. Gradient accumulation boundaries
3. Advanced checkpointing strategies
4. Learning rate scheduling with FP16
5. Prefetching and data pipeline optimization
6. Multi-phase training (different stages)

DIFFERENCES FROM SIMPLE EXAMPLES:
- Custom data provider (Bing's optimized pipeline)
- Manual gradient accumulation control
- Complex checkpoint management
- Production monitoring and logging
- Multi-node scaling optimizations
"""

import os
import time
import deepspeed
import torch
import torch.distributed as dist

from turing.models import BertMultiTask  # Microsoft's BERT implementation
from turing.dataset import PreTrainingDataset


# Global state (production pattern for tracking across function calls)
global_step = 0
global_data_samples = 0
last_global_step_from_restore = 0
all_step_time = 0.0


# ============================================================================
# CHECKPOINT MANAGEMENT (Production Pattern)
# ============================================================================

def checkpoint_model(PATH, ckpt_id, model, epoch, last_global_step,
                     last_global_data_samples, **kwargs):
    """
    [ANNOTATION] **PRODUCTION CHECKPOINTING**

    This is more sophisticated than simple torch.save():
    - Tracks global step (not just epoch)
    - Tracks data samples processed (for exact resumption)
    - Uses DeepSpeed's distributed checkpoint saving
    - Saves additional metadata (kwargs)

    Why track data samples?
    - Different nodes may process different amounts of data
    - Need to resume from exact data position
    - Ensures reproducibility across restarts
    """
    checkpoint_state_dict = {
        'epoch': epoch,
        'last_global_step': last_global_step,
        'last_global_data_samples': last_global_data_samples
    }

    # Add any additional state
    checkpoint_state_dict.update(kwargs)

    # [ANNOTATION] DeepSpeed's distributed checkpoint saving
    # Handles ZeRO partitioned states automatically
    success = model.network.save_checkpoint(PATH, ckpt_id, checkpoint_state_dict)

    status_msg = f'checkpointing: PATH={PATH}, ckpt_id={ckpt_id}'
    if success:
        logging.info(f"Success {status_msg}")
    else:
        logging.warning(f"Failure {status_msg}")

    return


def load_training_checkpoint(args, model, PATH, ckpt_id):
    """
    [ANNOTATION] **CHECKPOINT LOADING**

    Returns:
    - epoch: Which epoch to resume from
    - last_global_step: Which step to resume from
    - last_global_data_samples: Which data sample to resume from
    """
    logger = args.logger

    # DeepSpeed loads both model and optimizer states
    _, checkpoint_state_dict = model.network.load_checkpoint(PATH, ckpt_id)

    epoch = checkpoint_state_dict['epoch']
    last_global_step = checkpoint_state_dict['last_global_step']
    last_global_data_samples = checkpoint_state_dict['last_global_data_samples']

    del checkpoint_state_dict
    return (epoch, last_global_step, last_global_data_samples)


# ============================================================================
# CUSTOM DATASET PROVIDER (Production Pattern)
# ============================================================================

def get_dataloader(args, dataset, eval_set=False):
    """
    [ANNOTATION] **CUSTOM DATA PROVIDER**

    Production training uses custom data providers instead of simple DataLoader:
    - Prefetching for hiding data loading latency
    - Custom batching strategies
    - Optimized for large-scale training

    Note: Uses generator (x for x in ...) for memory efficiency
    """
    if args.local_rank == -1:
        train_sampler = RandomSampler(dataset)
    else:
        # [ANNOTATION] DistributedSampler for multi-GPU
        train_sampler = DistributedSampler(dataset)

    return (x for x in DataLoader(
        dataset,
        batch_size=args.train_micro_batch_size_per_gpu // 2 if eval_set
                   else args.train_micro_batch_size_per_gpu,
        sampler=train_sampler,
        num_workers=args.config['training']['num_workers']
    ))


# ============================================================================
# TRAINING FUNCTION WITH GRADIENT ACCUMULATION BOUNDARIES
# ============================================================================

def train(args, index, model, optimizer, pretrain_dataset_provider, finetune=False):
    """
    [ANNOTATION] **MAIN TRAINING LOOP WITH PRODUCTION PATTERNS**

    Key differences from simple training:
    1. Uses dataset provider (not simple dataloader)
    2. Manual gradient accumulation boundary checking
    3. Tracks global data samples (not just steps)
    4. Prefetching next shard while training
    5. Complex learning rate scheduling
    """
    global global_step
    global global_data_samples
    global last_global_step_from_restore
    global all_step_time

    # [ANNOTATION] Get data shard for this epoch
    # Production: Dataset sharded across epochs for efficiency
    dataset_iterator, total_length = pretrain_dataset_provider.get_shard(index)
    current_data_sample_count = global_data_samples

    config = args.config
    logger = args.logger

    logger.info(
        f'worker-{dist.get_rank()}: begin epoch {index+1} '
        f'current_sample_count {current_data_sample_count} '
        f'shard_length {total_length} '
        f'global_data_samples {global_data_samples}'
    )

    # [ANNOTATION] **PREFETCHING OPTIMIZATION**
    # While training on current shard, prefetch next shard
    # Hides data loading latency behind training
    pretrain_dataset_provider.prefetch_shard(index + 1)

    model.train()

    for _, batch_index in enumerate(tqdm(dataset_iterator, smoothing=1)):
        try:
            step_start = time.time()

            # [ANNOTATION] Get batch from custom provider
            batch = pretrain_dataset_provider.get_batch(batch_index)
            batch = tuple(t.to(args.device) for t in batch)

            # [ANNOTATION] **FORWARD PASS**
            loss = model.network(batch)
            unscaled_loss = loss.item()

            # Track data samples
            current_data_sample_count += (
                args.train_micro_batch_size_per_gpu * dist.get_world_size()
            )

            # [ANNOTATION] **PREFETCH NEXT BATCH**
            # While backward is running, prefetch next batch
            # Production optimization for data pipeline
            pretrain_dataset_provider.prefetch_batch()

            # [ANNOTATION] **BACKWARD PASS**
            model.network.backward(loss)
            loss = None  # Free memory

            # [ANNOTATION] **GRADIENT ACCUMULATION BOUNDARY CHECK**
            # This is the key pattern for gradient accumulation
            if model.network.is_gradient_accumulation_boundary():
                # We've accumulated enough gradients, time to update

                if args.fp16:
                    # [ANNOTATION] FP16 LEARNING RATE ADJUSTMENT
                    # With FP16, need to adjust LR manually after optimizer step
                    lr_this_step = update_learning_rate(
                        args, config, global_step, optimizer
                    )

                # Log metrics
                report_step_metrics(
                    args, lr_this_step, unscaled_loss,
                    global_step, current_data_sample_count
                )

                # [ANNOTATION] **OPTIMIZER STEP**
                # This is where actual parameter update happens
                # After gradient accumulation is complete
                model.network.step()

                # Report optimizer statistics (for LAMB optimizer)
                report_lamb_coefficients(args, optimizer)

                global_step += 1
                epoch_step += 1

            else:
                # [ANNOTATION] **MICRO-STEP (Gradient Accumulation)**
                # Just accumulate gradients, don't update parameters yet
                # Call step() to advance DeepSpeed's internal counters
                model.network.step()

            # [ANNOTATION] What is_gradient_accumulation_boundary() does:
            #
            # DeepSpeed tracks micro-steps internally:
            # - Micro-step 1: accumulate gradients → boundary() = False
            # - Micro-step 2: accumulate gradients → boundary() = False
            # - Micro-step 3: accumulate gradients → boundary() = False
            # - Micro-step 4: ready to update → boundary() = True
            #
            # Configuration:
            # {
            #   "train_micro_batch_size_per_gpu": 8,
            #   "gradient_accumulation_steps": 4
            # }
            #
            # Effective batch size: 8 × 4 × num_gpus

        except StopIteration:
            continue


# ============================================================================
# LEARNING RATE SCHEDULING (FP16 Production Pattern)
# ============================================================================

def update_learning_rate(args, config, global_step, optimizer):
    """
    [ANNOTATION] **FP16 LEARNING RATE SCHEDULING**

    When using FP16 with custom optimizer (not DeepSpeed's built-in):
    - Must manually update learning rate
    - Applies warmup schedule
    - Applies decay schedule

    This is production pattern for BERT pre-training:
    - Linear warmup for first N steps
    - Linear decay afterwards
    """
    # Get target learning rate based on schedule
    lr = get_learning_rate_scheduler(
        global_step,
        config['training']['learning_rate'],
        config['training']['warmup_proportion'],
        config['training']['total_training_steps']
    )

    # Update optimizer's learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


# ============================================================================
# MULTI-PHASE TRAINING (Production Pattern)
# ============================================================================

"""
PRODUCTION BERT TRAINING STRATEGY:
-----------------------------------

Phase 1: Short sequences (128 tokens), 90% of steps
  - Faster training (smaller sequences)
  - Learn basic language patterns
  - Batch size: Larger (more samples)

Phase 2: Long sequences (512 tokens), 10% of steps
  - Learn long-range dependencies
  - Full positional embeddings
  - Batch size: Smaller (memory constrained)

Example Schedule:
  Steps 0-90000: sequence_length=128, batch_size=4096
  Steps 90000-100000: sequence_length=512, batch_size=1024

Why this works:
  - Most language understanding happens at short range
  - Only need long sequences for final fine-tuning
  - Saves ~3× compute time
"""


# ============================================================================
# PREFETCHING STRATEGY (Production Optimization)
# ============================================================================

"""
PREFETCHING PIPELINE:
---------------------

Traditional (slow):
  [Load Batch 1] [Train Batch 1] [Load Batch 2] [Train Batch 2] ...
       ↑ GPU idle           ↑ I/O idle

With Prefetching:
  [Load Batch 1] [Train Batch 1] [Train Batch 2] [Train Batch 3] ...
                 [Load Batch 2]  [Load Batch 3]  [Load Batch 4]
                        ↑ Overlapped!

Implementation:
1. prefetch_shard(index + 1): Load next epoch's data while training current
2. prefetch_batch(): Load next batch while training current

Benefits:
  - Hides I/O latency behind computation
  - Keep GPU saturated
  - Critical for large-scale training (1024 GPUs!)
"""


# ============================================================================
# GRADIENT ACCUMULATION BENEFITS
# ============================================================================

"""
WHY GRADIENT ACCUMULATION?
---------------------------

Problem: Large batch sizes don't fit in memory
  - BERT-Large: ~340M parameters
  - Batch size 4096: Too large for single GPU

Solution: Accumulate gradients over multiple micro-batches

Example:
  Effective batch size: 4096
  Micro batch size: 32 (fits in GPU)
  Gradient accumulation steps: 128
  GPUs: 1

  Process:
    Forward-Backward micro-batch 1 (size 32) → accumulate gradients
    Forward-Backward micro-batch 2 (size 32) → accumulate gradients
    ...
    Forward-Backward micro-batch 128 (size 32) → accumulate gradients
    Optimizer step (effective batch size: 32 × 128 = 4096)

Multi-GPU:
  With 8 GPUs:
    Micro batch per GPU: 32
    Gradient accumulation: 16
    Effective batch: 32 × 16 × 8 = 4096

Benefits:
  ✓ Train with large batch sizes
  ✓ Improve convergence (large batch = better gradients)
  ✓ Better hardware utilization
  ✓ Matches batch sizes from papers
"""


# ============================================================================
# PRODUCTION MONITORING
# ============================================================================

def report_step_metrics(args, lr, loss, global_step, data_samples):
    """
    [ANNOTATION] **PRODUCTION MONITORING**

    Production training tracks:
    - Loss (for convergence monitoring)
    - Learning rate (verify schedule)
    - Throughput (samples/second)
    - Data samples processed (for checkpointing)
    - GPU memory usage
    - Communication time breakdown

    Critical for:
    - Detecting training issues early
    - Optimizing performance
    - Debugging distributed training
    """
    if global_step % args.log_interval == 0:
        logger.info(
            f'Step {global_step}: '
            f'loss={loss:.4f}, '
            f'lr={lr:.6f}, '
            f'samples={data_samples}'
        )

        # Log to TensorBoard (if enabled)
        if args.tensorboard_writer:
            args.tensorboard_writer.add_scalar('Loss/train', loss, global_step)
            args.tensorboard_writer.add_scalar('LR', lr, global_step)


# ============================================================================
# KEY TAKEAWAYS FOR PRODUCTION BERT TRAINING
# ============================================================================

"""
1. DATASET PROVIDER PATTERN:
   - Custom data provider (not simple DataLoader)
   - Prefetching for hiding I/O latency
   - Sharded dataset for distributed training
   - Optimized for large-scale

2. GRADIENT ACCUMULATION BOUNDARIES:
   - Manual control via is_gradient_accumulation_boundary()
   - Optimizer step only at boundaries
   - Micro-steps call step() but don't update
   - Critical for large effective batch sizes

3. CHECKPOINT MANAGEMENT:
   - Track global step, epoch, AND data samples
   - Enables exact resumption after failure
   - DeepSpeed handles distributed state
   - Save frequently (production reliability)

4. LEARNING RATE SCHEDULING:
   - Warmup for first 10% of steps
   - Linear decay afterwards
   - Manual update when using FP16
   - Critical for BERT convergence

5. MULTI-PHASE TRAINING:
   - Phase 1: Short sequences (90% of training)
   - Phase 2: Long sequences (10% of training)
   - Saves compute time
   - Production strategy from BERT paper

6. PREFETCHING:
   - Prefetch next shard while training current
   - Prefetch next batch during backward
   - Hides I/O latency
   - Essential for 1024-GPU scale

7. MONITORING:
   - Track loss, LR, throughput
   - TensorBoard integration
   - GPU memory monitoring
   - Critical for debugging at scale

8. DISTRIBUTED PATTERNS:
   - DistributedSampler for data sharding
   - Barrier synchronization points
   - Rank-0 only for logging/checkpointing
   - All-reduce for metric aggregation
"""


# ============================================================================
# RECORD ACHIEVEMENT: 44-MINUTE BERT
# ============================================================================

"""
MICROSOFT'S RECORD BERT TRAINING:
----------------------------------

Achievement: 44 minutes for BERT-Large pre-training
Hardware: 1024 V100 GPUs (128 DGX-2 nodes)
Configuration:
  - Model: BERT-Large (340M parameters)
  - Sequence Length: 128 → 512 (multi-phase)
  - Batch Size: 65,536 (via gradient accumulation)
  - Optimizer: LAMB (large-batch optimizer)
  - Precision: FP16 mixed precision

Key Optimizations:
1. LAMB Optimizer:
   - Designed for large-batch training
   - Layer-wise adaptive learning rates
   - Better convergence than Adam at large batch sizes

2. Gradient Accumulation:
   - Effective batch size: 65,536
   - Micro batch per GPU: 64
   - Accumulation steps: 1024 / (8 GPUs/node × 128 nodes × 64)

3. Multi-Phase Training:
   - 90% at sequence length 128
   - 10% at sequence length 512
   - Saves ~3× training time

4. Communication Optimization:
   - InfiniBand interconnect (200 Gb/s)
   - Optimized all-reduce (NCCL)
   - Gradient compression (optional)

5. Data Pipeline:
   - Prefetching (hide I/O latency)
   - Efficient data sharding
   - Custom dataset provider

Previous Record: ~67 hours (Google)
Microsoft: 44 minutes (90× faster!)
"""


# ============================================================================
# CONFIGURATION EXAMPLE
# ============================================================================

"""
# DeepSpeed config for production BERT training:

{
  "train_batch_size": 65536,  # Effective batch size
  "train_micro_batch_size_per_gpu": 64,
  "gradient_accumulation_steps": 8,  # Per GPU

  "optimizer": {
    "type": "Lamb",  # Large-batch optimizer
    "params": {
      "lr": 0.00176,
      "betas": [0.9, 0.999],
      "eps": 1e-6,
      "weight_decay": 0.01
    }
  },

  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 0.00176,
      "warmup_num_steps": 1000,
      "total_num_steps": 10000
    }
  },

  "gradient_clipping": 1.0,

  "fp16": {
    "enabled": true,
    "loss_scale": 0,  # Dynamic loss scaling
    "initial_scale_power": 16
  },

  "zero_optimization": {
    "stage": 1,  # Optimizer state partitioning
    "allgather_partitions": true,
    "reduce_scatter": true,
    "overlap_comm": true,
    "contiguous_gradients": true
  },

  "wall_clock_breakdown": true,  # Profile communication
  "steps_per_print": 100
}
"""


# ============================================================================
# USAGE
# ============================================================================

"""
# Launch on 128 nodes (1024 GPUs):

deepspeed --num_nodes=128 --num_gpus=8 \\
    --hostfile=hostfile \\
    deepspeed_train.py \\
    --data_path /data/wikipedia_bookcorpus \\
    --config_file bert_large_config.json \\
    --output_dir /output/bert_large \\
    --epochs 1 \\
    --checkpoint_dir /checkpoints

# Key parameters:
# - num_nodes: 128 (DGX-2 nodes)
# - num_gpus: 8 per node
# - Total GPUs: 128 × 8 = 1024
# - Training time: 44 minutes
# - Cost: ~$100 (on cloud)
"""

# [ANNOTATION] See training/bing_bert/deepspeed_train.py for full implementation
