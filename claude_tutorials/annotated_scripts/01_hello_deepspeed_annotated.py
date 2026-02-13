"""
ANNOTATED: HelloDeepSpeed - BERT MLM Training with DeepSpeed

Original File: training/HelloDeepSpeed/train_bert_ds.py

This script demonstrates the basic integration of DeepSpeed with PyTorch for distributed training.
It trains a BERT-style model on the Masked Language Modeling (MLM) task using the WikiText dataset.

KEY DEEPSPEED CONCEPTS DEMONSTRATED:
1. DeepSpeed initialization with config dictionary
2. ZeRO optimizer state partitioning (Stage 1)
3. Mixed precision training (FP16/BF16)
4. DeepSpeed engine's forward/backward/step API
5. DeepSpeed checkpointing

DISTRIBUTED TRAINING FLOW:
- No explicit `deepspeed.init_distributed()` call needed
- DeepSpeed launcher (e.g., `deepspeed train_bert_ds.py`) handles process group init
- Uses environment variables (RANK, LOCAL_RANK, WORLD_SIZE) set by launcher
"""

import os
import deepspeed  # [ANNOTATION] Import DeepSpeed library
from deepspeed.accelerator import get_accelerator  # [ANNOTATION] Hardware abstraction (GPU/CPU/NPU)

# ============================================================================
# DISTRIBUTED TRAINING UTILITY FUNCTIONS
# ============================================================================

def is_rank_0() -> bool:
    """
    [ANNOTATION] Check if current process is rank 0 (master process).

    DISTRIBUTED CONCEPT:
    - In distributed training, each GPU runs a separate process
    - Rank 0 is typically responsible for logging, checkpointing, and I/O
    - RANK environment variable is set by the DeepSpeed launcher

    WHEN TO USE:
    - Before printing logs (to avoid duplicate output from all ranks)
    - Before saving checkpoints (only one process should write to disk)
    - Before creating TensorBoard writers
    """
    return int(os.environ.get("RANK", "0")) == 0


# ============================================================================
# DEEPSPEED INITIALIZATION AND TRAINING (KEY SECTION)
# ============================================================================

def train(...):
    """Main training function with DeepSpeed integration."""

    # ------------------------------------------------------------------------
    # STEP 1: Device Setup
    # ------------------------------------------------------------------------
    # [ANNOTATION] Get the local GPU device for this process
    # local_rank is the GPU ID on the current machine (0-7 on an 8-GPU node)
    # This is different from global rank which is unique across all machines
    device = (torch.device(get_accelerator().device_name(), local_rank)
              if (local_rank > -1) and get_accelerator().is_available()
              else torch.device("cpu"))

    # ------------------------------------------------------------------------
    # STEP 2: Create Model (Standard PyTorch)
    # ------------------------------------------------------------------------
    model = create_model(
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        h_dim=h_dim,
        dropout=dropout,
    )
    # [ANNOTATION] Model is created on CPU first. DeepSpeed will handle device placement.

    # ------------------------------------------------------------------------
    # STEP 3: DeepSpeed Configuration
    # ------------------------------------------------------------------------
    # [ANNOTATION] **CRITICAL**: DeepSpeed configuration dictionary
    # This controls all ZeRO optimizations and training behavior

    ds_config = {
        # Batch size per GPU. Total batch size = this * num_gpus * gradient_accumulation_steps
        "train_micro_batch_size_per_gpu": batch_size,

        # Optimizer configuration
        # DeepSpeed will create the optimizer internally based on this config
        "optimizer": {
            "type": "Adam",  # DeepSpeed has optimized Adam implementations
            "params": {
                "lr": 1e-4
            }
        },

        # Mixed precision training (FP16 or BF16)
        # This section is dynamically set based on the 'dtype' argument
        dtype: {
            "enabled": True
        },

        # **ZeRO OPTIMIZATION CONFIGURATION** (Most Important Section)
        "zero_optimization": {
            "stage": 1,  # ZeRO Stage 1 = Optimizer State Partitioning

            # CPU Offloading: Move optimizer states to CPU to save GPU memory
            "offload_optimizer": {
                "device": "cpu"  # Offload optimizer states to CPU RAM
            }
        }
    }

    # [ANNOTATION] ZeRO Stage Explanation:
    # - Stage 0: Disabled (standard data parallelism, full optimizer state on each GPU)
    # - Stage 1: Partition optimizer states across GPUs (this example)
    # - Stage 2: Partition optimizer states + gradients across GPUs
    # - Stage 3: Partition optimizer states + gradients + model parameters across GPUs

    # ------------------------------------------------------------------------
    # STEP 4: DeepSpeed Initialization
    # ------------------------------------------------------------------------
    # [ANNOTATION] **CRITICAL API CALL**: deepspeed.initialize()
    # This is where PyTorch model becomes a DeepSpeed model

    model, _, _, _ = deepspeed.initialize(
        model=model,                          # PyTorch model
        model_parameters=model.parameters(),  # Model parameters for optimizer
        config=ds_config                      # DeepSpeed config dict (defined above)
    )

    # [ANNOTATION] What happens inside deepspeed.initialize():
    # 1. Initializes distributed process group (if not already initialized)
    # 2. Moves model to appropriate device (GPU/CPU)
    # 3. Wraps model with DeepSpeedEngine
    # 4. Creates DeepSpeed optimizer based on config
    # 5. Sets up ZeRO partitioning (if stage > 0)
    # 6. Configures mixed precision (FP16/BF16)
    # 7. Sets up gradient clipping, learning rate scheduling, etc.

    # [ANNOTATION] Return values explained:
    # - model: DeepSpeedEngine (wraps your PyTorch model)
    # - optimizer: DeepSpeed optimizer (managed internally)
    # - _, _: Training dataloader and LR scheduler (not used here)

    # ------------------------------------------------------------------------
    # STEP 5: Load Checkpoint (Optional)
    # ------------------------------------------------------------------------
    start_step = 1
    if load_checkpoint_dir is not None:
        # [ANNOTATION] DeepSpeed's built-in checkpointing
        # Automatically handles ZeRO partitioned states
        _, client_state = model.load_checkpoint(load_dir=load_checkpoint_dir)
        checkpoint_step = client_state['checkpoint_step']
        start_step = checkpoint_step + 1

    # ------------------------------------------------------------------------
    # STEP 6: Training Loop
    # ------------------------------------------------------------------------
    model.train()
    losses = []
    for step, batch in enumerate(data_iterator, start=start_step):
        if step >= num_iterations:
            break

        # Move batch to device
        for key, value in batch.items():
            batch[key] = value.to(device)

        # [ANNOTATION] **FORWARD PASS**
        # Call the DeepSpeed model like a normal PyTorch model
        loss = model(**batch)

        # [ANNOTATION] What happens in forward pass with ZeRO:
        # - Stage 1: Model weights are already on GPU (same as standard training)
        # - Stage 2: Model weights are already on GPU, gradients will be partitioned
        # - Stage 3: Parameters are gathered (All-Gather) before each layer's computation

        # [ANNOTATION] **BACKWARD PASS**
        # Use DeepSpeed's backward method instead of loss.backward()
        model.backward(loss)

        # [ANNOTATION] What happens in backward pass with ZeRO:
        # - Computes gradients
        # - Stage 1: Gradients are kept full on each GPU
        # - Stage 2: Gradients are partitioned and scattered to corresponding GPU
        # - Stage 3: Parameters are released after use (reduce memory)
        # - If CPU offload is enabled, optimizer states are on CPU

        # [ANNOTATION] **OPTIMIZER STEP**
        # Use DeepSpeed's step method instead of optimizer.step()
        model.step()

        # [ANNOTATION] What happens in optimizer step with ZeRO:
        # - Each GPU updates only its partition of optimizer states
        # - Stage 1: Each GPU has 1/N of optimizer states
        # - If CPU offload: Optimizer states are updated on CPU
        # - Updated parameters are synchronized across GPUs

        losses.append(loss.item())

        # Logging (rank 0 only)
        if step % log_every == 0:
            log_dist("Loss: {0:.4f}".format(np.mean(losses)),
                     ranks=[0],
                     level=logging.INFO)

        # [ANNOTATION] **CHECKPOINTING**
        if step % checkpoint_every == 0:
            # DeepSpeed's save_checkpoint handles ZeRO partitioned states
            model.save_checkpoint(
                save_dir=exp_dir,
                client_state={'checkpoint_step': step}
            )
            # [ANNOTATION] What gets saved:
            # - Model parameters (gathered from all GPUs if ZeRO-3)
            # - Optimizer states (partitioned across GPUs, saved accordingly)
            # - Learning rate scheduler state
            # - Custom client_state (checkpoint_step in this case)

    return exp_dir


# ============================================================================
# COMMAND LINE EXECUTION
# ============================================================================

# [ANNOTATION] How to run this script:
#
# Single GPU:
#   deepspeed --num_gpus=1 train_bert_ds.py --checkpoint_dir ./checkpoints
#
# Multi-GPU (single node):
#   deepspeed --num_gpus=4 train_bert_ds.py --checkpoint_dir ./checkpoints
#
# Multi-node (e.g., 2 nodes with 8 GPUs each):
#   deepspeed --num_nodes=2 --num_gpus=8 train_bert_ds.py --checkpoint_dir ./checkpoints
#
# With custom hostfile:
#   deepspeed --hostfile=myhostfile train_bert_ds.py --checkpoint_dir ./checkpoints
#
# [ANNOTATION] The DeepSpeed launcher:
# - Sets environment variables: RANK, LOCAL_RANK, WORLD_SIZE
# - Initializes the distributed backend (NCCL for GPU, Gloo for CPU)
# - Launches one process per GPU
# - Handles inter-node communication setup


# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

"""
1. INITIALIZATION:
   - Use deepspeed.initialize() instead of manual model.to(device) and optimizer creation
   - DeepSpeed config dict controls all optimizations
   - Launcher handles distributed setup automatically

2. TRAINING LOOP CHANGES:
   - model(**batch) instead of model.forward()
   - model.backward(loss) instead of loss.backward()
   - model.step() instead of optimizer.step() and optimizer.zero_grad()

3. CHECKPOINTING:
   - Use model.save_checkpoint() and model.load_checkpoint()
   - Automatically handles ZeRO partitioned states
   - No need to manually gather/scatter weights

4. CONFIGURATION:
   - "zero_optimization.stage" controls memory optimization level
   - "offload_optimizer.device" enables CPU offloading
   - "train_micro_batch_size_per_gpu" sets per-GPU batch size

5. DISTRIBUTED CONCEPTS:
   - Use is_rank_0() for single-process operations
   - DeepSpeed handles all inter-GPU communication
   - No need to manually use torch.distributed APIs
"""


# ============================================================================
# DEEPSPEED VS STANDARD PYTORCH: CODE COMPARISON
# ============================================================================

"""
STANDARD PYTORCH DISTRIBUTED:
------------------------------
import torch.distributed as dist
dist.init_process_group(backend='nccl')
model = Model().to(device)
model = torch.nn.parallel.DistributedDataParallel(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


DEEPSPEED VERSION:
------------------
import deepspeed
ds_config = {
    "train_micro_batch_size_per_gpu": 8,
    "optimizer": {"type": "Adam", "params": {"lr": 1e-4}},
    "zero_optimization": {"stage": 1}
}
model, optimizer, _, _ = deepspeed.initialize(
    model=Model(),
    model_parameters=Model().parameters(),
    config=ds_config
)

for batch in dataloader:
    loss = model(batch)
    model.backward(loss)
    model.step()

KEY DIFFERENCES:
- No explicit optimizer creation (DeepSpeed creates it)
- No manual zero_grad() (DeepSpeed handles it in step())
- No explicit DDP wrapping (DeepSpeed wraps automatically)
- Automatic ZeRO optimizations based on config
"""

# [ANNOTATION] See training/HelloDeepSpeed/train_bert_ds.py for full implementation
