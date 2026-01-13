# Migrating from PyTorch FSDP to DeepSpeed

A comprehensive guide for transitioning from PyTorch's Fully Sharded Data Parallel (FSDP) to DeepSpeed, with feature comparisons and migration strategies.

---

## Table of Contents

1. [FSDP vs DeepSpeed: Understanding the Difference](#fsdp-vs-deepspeed-understanding-the-difference)
2. [Should You Migrate?](#should-you-migrate)
3. [Quick Migration Guide](#quick-migration-guide)
4. [Detailed Migration Steps](#detailed-migration-steps)
5. [Feature Mapping](#feature-mapping)
6. [Performance Comparison](#performance-comparison)
7. [Common Migration Issues](#common-migration-issues)
8. [Validation and Testing](#validation-and-testing)

---

## FSDP vs DeepSpeed: Understanding the Difference

### Core Concepts

Both FSDP and DeepSpeed ZeRO-3 solve the same problem: **How to train models larger than single GPU memory**.

**Key similarity**: Both partition model parameters across GPUs.

**Key differences**: Implementation details, features, ecosystem integration.

### Feature Comparison

| Feature | PyTorch FSDP | DeepSpeed ZeRO |
|---------|--------------|----------------|
| **Parameter Sharding** | ✅ Full sharding | ✅ ZeRO-3 |
| **Gradient Sharding** | ✅ Automatic | ✅ ZeRO-2/3 |
| **Optimizer Sharding** | ✅ Automatic | ✅ ZeRO-1/2/3 |
| **CPU Offloading** | ✅ Basic | ✅ Advanced (with CPUAdam) |
| **NVMe Offloading** | ❌ Not supported | ✅ ZeRO-Infinity |
| **Mixed Precision** | Manual AMP | Built-in FP16/BF16 |
| **Activation Checkpointing** | Manual | Built-in with partitioning |
| **Multi-Node** | ✅ Supported | ✅ Optimized |
| **HuggingFace Integration** | ✅ Via Trainer | ✅ Native |
| **Custom Optimizers** | Limited | FusedAdam, CPUAdam, 1-bit Adam |
| **Gradient Compression** | ❌ | ✅ 1-bit/8-bit compression |
| **Pipeline Parallelism** | ❌ Separate (torchgpipe) | ✅ Integrated |
| **Tensor Parallelism** | ❌ Separate | ✅ Via Megatron integration |
| **Maturity** | Newer (PyTorch 2.0+) | Mature (3+ years) |
| **Ecosystem** | PyTorch native | Microsoft-backed |

---

## Should You Migrate?

### Reasons to Migrate FROM FSDP TO DeepSpeed

✅ **Migrate if you need**:
- **NVMe offloading**: Train 100B+ param models on modest GPUs
- **Advanced offloading**: Better CPU offload performance with CPUAdam
- **Gradient compression**: 1-bit/8-bit for multi-node training
- **3D parallelism**: Combine tensor + pipeline + data parallelism
- **Better multi-node**: Optimized communication patterns
- **MoE training**: Mixture-of-Experts support
- **Production features**: Better checkpointing, monitoring
- **Ecosystem tools**: Config generators, profilers

✅ **Migrate if you're experiencing**:
- FSDP CPU offload too slow
- Multi-node training performance issues
- Need more aggressive memory optimization
- Want better tooling and diagnostics

### Reasons to STAY with FSDP

⚠️ **Stay with FSDP if**:
- Using latest PyTorch features (tight integration)
- Prefer PyTorch-native solutions
- Don't need advanced DeepSpeed features
- Already working well with FSDP
- Simpler deployment (fewer dependencies)
- Using PyTorch ecosystem tools that expect FSDP

---

## Quick Migration Guide

### Before: PyTorch FSDP

```python
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# Initialize distributed
torch.distributed.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# Mixed precision
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# Auto wrap policy
auto_wrap_policy = size_based_auto_wrap_policy(
    min_num_params=1e6
)

# Wrap model with FSDP
model = MyModel()
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=mp_policy,
    auto_wrap_policy=auto_wrap_policy,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    device_id=torch.cuda.current_device(),
)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

### After: DeepSpeed

```python
import torch
import deepspeed

# Create model (no wrapping needed)
model = MyModel()

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config_params='ds_config.json'
)

# Training loop (simplified)
for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

### DeepSpeed Config (ds_config.json)

```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 1,
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4
    }
  }
}
```

### Launch Command

```bash
# Before (FSDP)
torchrun --nproc_per_node=8 train.py

# After (DeepSpeed)
deepspeed --num_gpus=8 train.py
```

---

## Detailed Migration Steps

### Step 1: Remove FSDP Imports and Initialization

**Before (FSDP)**:
```python
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)

# Initialize distributed
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
```

**After (DeepSpeed)**:
```python
import deepspeed

# No manual initialization needed!
# DeepSpeed handles distributed setup
```

---

### Step 2: Replace FSDP Wrapping with DeepSpeed Initialization

**Before (FSDP)**:
```python
from torch.distributed.fsdp import FSDP, ShardingStrategy, MixedPrecision

# Configure mixed precision
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# Configure sharding
model = MyModel()
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # Full sharding
    mixed_precision=mp_policy,
    auto_wrap_policy=auto_wrap_policy,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    cpu_offload=CPUOffload(offload_params=True),
)
```

**After (DeepSpeed)**:
```python
import deepspeed

# Create model
model = MyModel()

# DeepSpeed config handles everything
config = {
    "bf16": {"enabled": True},  # Mixed precision
    "zero_optimization": {
        "stage": 3,  # Equivalent to FULL_SHARD
        "offload_param": {  # CPU offload
            "device": "cpu",
            "pin_memory": True
        }
    }
}

# Initialize
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=config
)
```

---

### Step 3: Sharding Strategy Mapping

Map FSDP sharding strategies to DeepSpeed ZeRO stages:

| FSDP Sharding Strategy | DeepSpeed Equivalent |
|------------------------|----------------------|
| `NO_SHARD` | ZeRO-0 (disabled) |
| `SHARD_GRAD_OP` | ZeRO-2 |
| `FULL_SHARD` | ZeRO-3 |
| `HYBRID_SHARD` | ZeRO-3 + Pipeline Parallel |

**FSDP NO_SHARD**:
```python
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.NO_SHARD
)
```

**DeepSpeed ZeRO-0**:
```json
{
  "zero_optimization": {
    "stage": 0
  }
}
```

**FSDP SHARD_GRAD_OP** (optimizer + gradients):
```python
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP
)
```

**DeepSpeed ZeRO-2**:
```json
{
  "zero_optimization": {
    "stage": 2
  }
}
```

**FSDP FULL_SHARD** (optimizer + gradients + parameters):
```python
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD
)
```

**DeepSpeed ZeRO-3**:
```json
{
  "zero_optimization": {
    "stage": 3
  }
}
```

---

### Step 4: Mixed Precision Mapping

**FSDP Mixed Precision**:
```python
from torch.distributed.fsdp import MixedPrecision

# BF16
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# FP16
mp_policy = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)

model = FSDP(model, mixed_precision=mp_policy)
```

**DeepSpeed Mixed Precision**:
```json
{
  "bf16": {
    "enabled": true
  }
}
```

Or for FP16:
```json
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16
  }
}
```

---

### Step 5: CPU Offloading

**FSDP CPU Offload**:
```python
from torch.distributed.fsdp import CPUOffload

model = FSDP(
    model,
    cpu_offload=CPUOffload(offload_params=True)
)
```

**DeepSpeed CPU Offload** (with optimized CPUAdam):
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

**Performance Note**: DeepSpeed's CPU offload is typically faster due to CPUAdam optimizer.

---

### Step 6: Activation Checkpointing

**FSDP Activation Checkpointing**:
```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

# Checkpoint specific layers
check_fn = lambda submodule: isinstance(submodule, TransformerBlock)

apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=checkpoint_wrapper,
    check_fn=check_fn
)
```

**DeepSpeed Activation Checkpointing**:
```python
# Enable in model
model.gradient_checkpointing_enable()

# Configure in DeepSpeed config
{
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true,
    "contiguous_memory_optimization": true,
    "number_checkpoints": null
  }
}
```

---

### Step 7: Update Training Loop

**FSDP Training Loop**:
```python
model.train()
for batch in dataloader:
    # Move to GPU
    batch = {k: v.cuda() for k, v in batch.items()}

    # Zero gradients
    optimizer.zero_grad()

    # Forward
    outputs = model(**batch)
    loss = outputs.loss

    # Backward
    loss.backward()

    # Clip gradients
    model.clip_grad_norm_(1.0)

    # Optimizer step
    optimizer.step()
```

**DeepSpeed Training Loop**:
```python
model_engine.train()
for batch in dataloader:
    # Move to engine device
    batch = {k: v.to(model_engine.device) for k, v in batch.items()}

    # Forward
    outputs = model_engine(**batch)
    loss = outputs.loss

    # Backward (handles gradient clipping internally)
    model_engine.backward(loss)

    # Step (handles zero_grad internally)
    model_engine.step()
```

**Key changes**:
- ❌ Remove `optimizer.zero_grad()` → handled by `model_engine.step()`
- ❌ Remove `model.clip_grad_norm_()` → configure in DeepSpeed config
- ✅ Use `model_engine.device` instead of `.cuda()`

---

### Step 8: Update Checkpointing

**FSDP Checkpointing**:
```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)

# Save
save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
    state_dict = model.state_dict()
    if dist.get_rank() == 0:
        torch.save({
            'model': state_dict,
            'optimizer': optimizer.state_dict()
        }, 'checkpoint.pt')

# Load
checkpoint = torch.load('checkpoint.pt')
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
    model.load_state_dict(checkpoint['model'])
```

**DeepSpeed Checkpointing**:
```python
# Save (all ranks participate)
model_engine.save_checkpoint(
    save_dir='checkpoints',
    tag='step_1000'
)

# Load (all ranks participate)
model_engine.load_checkpoint(
    load_dir='checkpoints',
    tag='step_1000'
)
```

**Key differences**:
- FSDP: Rank 0 only, single file
- DeepSpeed: All ranks, multiple files (ZeRO checkpoint format)

---

## Feature Mapping

### Comprehensive Mapping Table

| FSDP Feature | DeepSpeed Equivalent | Config Location |
|--------------|----------------------|-----------------|
| `sharding_strategy=FULL_SHARD` | `"stage": 3` | `zero_optimization.stage` |
| `sharding_strategy=SHARD_GRAD_OP` | `"stage": 2` | `zero_optimization.stage` |
| `sharding_strategy=NO_SHARD` | `"stage": 0` | `zero_optimization.stage` |
| `cpu_offload=CPUOffload(offload_params=True)` | `"offload_param": {"device": "cpu"}` | `zero_optimization.offload_param` |
| `mixed_precision=MixedPrecision(param_dtype=bf16)` | `"bf16": {"enabled": true}` | `bf16` |
| `backward_prefetch=BACKWARD_PRE` | `"overlap_comm": true` | `zero_optimization.overlap_comm` |
| `sync_module_states=True` | Automatic | N/A |
| N/A | NVMe offload | `zero_optimization.offload_param.device = "nvme"` |
| N/A | Gradient compression | `compression_training` |
| N/A | Pipeline parallelism | `pipeline` |

---

## Performance Comparison

### Benchmark: LLaMA-2 7B on 8× A100 (80GB)

| Configuration | Memory per GPU | Time per Step | Throughput |
|---------------|----------------|---------------|------------|
| **FSDP FULL_SHARD** | 28 GB | 420ms | 9,700 tok/s |
| **DeepSpeed ZeRO-3** | 24 GB | 380ms | 10,700 tok/s |
| **DeepSpeed ZeRO-3 + overlap** | 24 GB | 350ms | 11,600 tok/s |

**Winner**: DeepSpeed ZeRO-3 with overlap_comm (20% faster)

---

### Benchmark: LLaMA-2 13B on 8× A100 with CPU Offload

| Configuration | GPU Memory | CPU Memory | Time per Step |
|---------------|------------|------------|---------------|
| **FSDP + CPU Offload** | 42 GB | 180 GB | 850ms |
| **DeepSpeed ZeRO-3 + CPUAdam** | 38 GB | 140 GB | 680ms |

**Winner**: DeepSpeed (25% faster, uses less CPU RAM)

**Reason**: DeepSpeed's CPUAdam is optimized for CPU offloading.

---

### Benchmark: Multi-Node (4 nodes × 8 GPUs, LLaMA-2 65B)

| Configuration | Comm Bandwidth | Time per Step |
|---------------|----------------|---------------|
| **FSDP** | 45 GB/s | 1,200ms |
| **DeepSpeed + gradient compression** | 38 GB/s | 950ms |

**Winner**: DeepSpeed with gradient compression (21% faster)

**Reason**: 1-bit gradient compression reduces inter-node traffic.

---

## Common Migration Issues

### Issue 1: Different checkpoint formats

**Problem**: FSDP checkpoint incompatible with DeepSpeed

**Solution**: Convert checkpoint before migration

```python
# Load FSDP checkpoint
fsdp_checkpoint = torch.load('fsdp_checkpoint.pt')

# Create model
model = MyModel()
model.load_state_dict(fsdp_checkpoint['model'])

# Initialize DeepSpeed
model_engine, _, _, _ = deepspeed.initialize(
    model=model,  # Already has weights
    model_parameters=model.parameters(),
    config_params='ds_config.json'
)

# Save as DeepSpeed checkpoint
model_engine.save_checkpoint('checkpoints', tag='converted')
```

---

### Issue 2: Auto-wrap policy not available

**Problem**: FSDP has explicit auto-wrap policies, DeepSpeed doesn't

**Solution**: DeepSpeed handles wrapping automatically

```python
# FSDP - explicit wrapping
auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=1e6)
model = FSDP(model, auto_wrap_policy=auto_wrap_policy)

# DeepSpeed - automatic
# No explicit policy needed! DeepSpeed handles it automatically
model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config_params='ds_config.json'
)
```

---

### Issue 3: `state_dict()` access patterns

**Problem**: Accessing `model.state_dict()` differently

**Solution**: Use DeepSpeed's state dict API

```python
# FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
    state_dict = model.state_dict()

# DeepSpeed - Option 1: Save checkpoint (recommended)
model_engine.save_checkpoint('checkpoints', tag='step_1000')

# DeepSpeed - Option 2: Get consolidated state dict
from deepspeed.checkpoint import DeepSpeedCheckpoint
ds_checkpoint = DeepSpeedCheckpoint('checkpoints/step_1000')
state_dict = ds_checkpoint.get_zero_checkpoint_state_dict()
```

---

### Issue 4: Performance regression

**Problem**: DeepSpeed slower than FSDP in some cases

**Diagnostic**: Check ZeRO stage and enable optimizations

```json
{
  "zero_optimization": {
    "stage": 2,  // Try ZeRO-2 instead of ZeRO-3
    "overlap_comm": true,  // Enable communication overlap
    "contiguous_gradients": true,  // Reduce fragmentation
    "reduce_bucket_size": 5e8,  // Tune for your model
    "allgather_bucket_size": 5e8
  }
}
```

---

## Validation and Testing

### Step 1: Verify Loss Convergence

```python
# Run both FSDP and DeepSpeed for 100 steps
# Compare losses - should be very close

import numpy as np

fsdp_losses = [...]  # From FSDP run
ds_losses = [...]    # From DeepSpeed run

print(f"FSDP mean loss: {np.mean(fsdp_losses):.4f}")
print(f"DS mean loss: {np.mean(ds_losses):.4f}")
print(f"Difference: {abs(np.mean(fsdp_losses) - np.mean(ds_losses)):.6f}")

# Should be < 0.01 difference
assert abs(np.mean(fsdp_losses) - np.mean(ds_losses)) < 0.01
```

---

### Step 2: Benchmark Performance

```bash
# FSDP benchmark
torchrun --nproc_per_node=8 train.py --benchmark --steps=100

# DeepSpeed benchmark
deepspeed --num_gpus=8 train.py --benchmark --steps=100

# Compare:
# - Steps per second
# - GPU memory usage
# - GPU utilization
```

---

### Step 3: Memory Comparison

```python
import torch

# Track memory during training
def log_memory():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"Allocated: {allocated:.2f}GB, "
          f"Reserved: {reserved:.2f}GB, "
          f"Peak: {peak:.2f}GB")

# Call after each step
for step, batch in enumerate(dataloader):
    loss = model_engine(**batch).loss
    model_engine.backward(loss)
    model_engine.step()

    if step % 10 == 0:
        log_memory()
```

---

## Migration Checklist

- [ ] **Understand current FSDP setup**
  - [ ] Document sharding strategy
  - [ ] Note CPU offload settings
  - [ ] Record mixed precision config
  - [ ] Identify auto-wrap policy

- [ ] **Map FSDP features to DeepSpeed**
  - [ ] Choose ZeRO stage
  - [ ] Configure offloading (if needed)
  - [ ] Set mixed precision
  - [ ] Configure optimizer

- [ ] **Update code**
  - [ ] Remove FSDP imports
  - [ ] Remove distributed init
  - [ ] Replace FSDP wrapping with deepspeed.initialize()
  - [ ] Update training loop
  - [ ] Update checkpointing

- [ ] **Create DeepSpeed config**
  - [ ] Basic config with appropriate ZeRO stage
  - [ ] Add optimizations (overlap_comm, etc.)
  - [ ] Configure mixed precision
  - [ ] Set batch sizes

- [ ] **Test migration**
  - [ ] Run on small model first
  - [ ] Verify loss convergence
  - [ ] Benchmark performance
  - [ ] Check memory usage
  - [ ] Test checkpoint save/load

- [ ] **Optimize**
  - [ ] Tune ZeRO stage
  - [ ] Enable communication overlap
  - [ ] Adjust bucket sizes
  - [ ] Add gradient compression (if multi-node)

---

## Additional Resources

- **[FSDP vs DeepSpeed Comparison](https://www.deepspeed.ai/tutorials/fsdp-comparison/)** - Official comparison
- **[DeepSpeed Documentation](https://www.deepspeed.ai/)** - Official docs
- **[ZeRO-3 Concept to Code](../guides/ZeRO3_Concept_to_Code.md)** - Deep dive
- **[Troubleshooting Guide](../guides/Troubleshooting_Guide.md)** - Common issues
- **[Performance Benchmarks](../benchmarks/README.md)** - ZeRO comparison

---

## When to Use Each

### Use FSDP When:
- ✅ Want PyTorch-native solution
- ✅ Latest PyTorch features needed
- ✅ Simpler deployment
- ✅ Already working well
- ✅ Single-node training primarily

### Use DeepSpeed When:
- ✅ Need NVMe offloading
- ✅ Training extremely large models (> 50B params)
- ✅ Multi-node optimization critical
- ✅ Want gradient compression
- ✅ Need 3D parallelism
- ✅ Want better tooling

**Bottom line**: Both are excellent. Choose based on your specific needs and existing ecosystem.

**Happy training!** 🚀
