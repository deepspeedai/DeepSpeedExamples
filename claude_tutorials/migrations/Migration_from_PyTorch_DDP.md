# Migrating from PyTorch DDP to DeepSpeed

A comprehensive guide for transitioning from PyTorch's DistributedDataParallel (DDP) to DeepSpeed, with side-by-side code comparisons and migration strategies.

---

## Table of Contents

1. [Why Migrate to DeepSpeed?](#why-migrate-to-deepspeed)
2. [Quick Migration (5 Minutes)](#quick-migration-5-minutes)
3. [Detailed Migration Steps](#detailed-migration-steps)
4. [Side-by-Side Code Comparison](#side-by-side-code-comparison)
5. [Configuration Mapping](#configuration-mapping)
6. [Performance Optimization](#performance-optimization)
7. [Common Migration Issues](#common-migration-issues)
8. [Validation and Testing](#validation-and-testing)

---

## Why Migrate to DeepSpeed?

### Key Benefits

| Feature | PyTorch DDP | DeepSpeed ZeRO |
|---------|-------------|----------------|
| **Memory Efficiency** | Replicates full model on each GPU | Partitions model across GPUs (up to 64× reduction) |
| **Max Model Size** | Limited by single GPU memory | Linear scaling with number of GPUs |
| **Optimizer** | Standard PyTorch optimizers | Optimized kernels (CPUAdam, FusedAdam) |
| **Offloading** | Not supported | CPU and NVMe offloading |
| **Activation Checkpointing** | Manual implementation | Built-in with partition support |
| **Mixed Precision** | Manual AMP setup | Automatic FP16/BF16 with loss scaling |
| **Gradient Accumulation** | Manual implementation | Built-in with proper memory handling |

### When to Migrate

✅ **Migrate if**:
- Running out of GPU memory
- Need to train larger models
- Want to reduce training costs
- Need better memory efficiency
- Training on multiple nodes

⚠️ **Consider staying with DDP if**:
- Model comfortably fits in single GPU
- Using very small models (< 500M parameters)
- Need maximum simplicity
- Don't need advanced features

---

## Quick Migration (5 Minutes)

### Before: PyTorch DDP

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# Create model
model = MyModel().cuda()
model = DDP(model, device_ids=[local_rank])

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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

# Create model (no .cuda() yet)
model = MyModel()

# Initialize DeepSpeed (replaces DDP + optimizer + AMP)
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

### Configuration File (ds_config.json)

```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-4
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2
  }
}
```

### Launch Command

```bash
# Before (DDP)
torchrun --nproc_per_node=8 train.py

# After (DeepSpeed)
deepspeed --num_gpus=8 train.py
```

---

## Detailed Migration Steps

### Step 1: Remove DDP Initialization

**Before (DDP)**:
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize distributed
dist.init_process_group(
    backend='nccl',
    init_method='env://'
)

local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# Wrap model
model = MyModel().cuda(local_rank)
model = DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=False
)
```

**After (DeepSpeed)**:
```python
import deepspeed

# No manual initialization needed!
# DeepSpeed handles everything

# Create model (don't move to cuda yet)
model = MyModel()

# DeepSpeed will handle device placement
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config_params='ds_config.json'
)
```

**Key Changes**:
- ❌ Remove `torch.distributed.init_process_group()`
- ❌ Remove `DDP()` wrapper
- ❌ Remove manual `.cuda()` calls
- ✅ Add `deepspeed.initialize()`

---

### Step 2: Replace Optimizer

**Before (DDP)**:
```python
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Create optimizer
optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

# Create scheduler
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=1000,
    eta_min=1e-6
)
```

**After (DeepSpeed - Method 1: Config)**:
```python
# Define in ds_config.json (recommended)
{
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 1e-6,
      "warmup_max_lr": 1e-4,
      "warmup_num_steps": 1000
    }
  }
}

# Initialize returns optimizer
model_engine, optimizer, _, scheduler = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config_params='ds_config.json'
)
```

**After (DeepSpeed - Method 2: Bring Your Own)**:
```python
# Create your own optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Pass to DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,  # Your optimizer
    config_params='ds_config.json'
)
```

**Key Changes**:
- ✅ Option 1: Define optimizer in config (simpler, recommended)
- ✅ Option 2: Pass existing optimizer to `deepspeed.initialize()`
- ⚠️ DeepSpeed may replace with optimized version (e.g., FusedAdam)

---

### Step 3: Update Training Loop

**Before (DDP)**:
```python
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        # Move batch to GPU
        batch = {k: v.cuda() for k, v in batch.items()}

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        # Scheduler step
        scheduler.step()
```

**After (DeepSpeed)**:
```python
model_engine.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        # Move batch to engine's device
        batch = {k: v.to(model_engine.device) for k, v in batch.items()}

        # Forward pass
        outputs = model_engine(**batch)
        loss = outputs.loss

        # Backward pass (handles gradient clipping internally)
        model_engine.backward(loss)

        # Optimizer step (handles scheduler internally)
        model_engine.step()
```

**Key Changes**:
- ❌ Remove `optimizer.zero_grad()` → handled by `model_engine.step()`
- ❌ Remove `loss.backward()` → use `model_engine.backward(loss)`
- ❌ Remove `optimizer.step()` → use `model_engine.step()`
- ❌ Remove `scheduler.step()` → handled internally if defined in config
- ❌ Remove manual gradient clipping → set in config
- ✅ Use `model_engine.device` instead of `local_rank`

---

### Step 4: Update Mixed Precision

**Before (DDP with PyTorch AMP)**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # Forward with autocast
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss

    # Backward with scaler
    scaler.scale(loss).backward()

    # Unscale and clip gradients
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Step with scaler
    scaler.step(optimizer)
    scaler.update()
```

**After (DeepSpeed)**:
```python
# Configure in ds_config.json
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,  // Dynamic loss scaling
    "initial_scale_power": 16,
    "loss_scale_window": 1000
  },
  "gradient_clipping": 1.0
}

# Training loop (no manual AMP code!)
for batch in dataloader:
    loss = model_engine(**batch).loss
    model_engine.backward(loss)
    model_engine.step()
```

**Or use BF16**:
```json
{
  "bf16": {
    "enabled": true  // More stable, if supported
  }
}
```

**Key Changes**:
- ❌ Remove all `autocast()` and `GradScaler()` code
- ✅ Enable FP16/BF16 in config
- ✅ DeepSpeed handles loss scaling automatically

---

### Step 5: Update Gradient Accumulation

**Before (DDP)**:
```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    # Forward
    loss = model(**batch).loss

    # Scale loss
    loss = loss / accumulation_steps

    # Backward
    loss.backward()

    # Step every N batches
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**After (DeepSpeed)**:
```json
{
  "train_batch_size": 128,  // Total effective batch size
  "train_micro_batch_size_per_gpu": 4,  // Per GPU per step
  "gradient_accumulation_steps": 32  // Auto-computed: 128 / (4 * num_gpus)
}
```

```python
# Training loop (no manual accumulation logic!)
for batch in dataloader:
    loss = model_engine(**batch).loss
    model_engine.backward(loss)
    model_engine.step()  // Handles accumulation automatically
```

**Key Changes**:
- ❌ Remove manual accumulation logic
- ❌ Remove loss scaling by accumulation steps
- ✅ Set `gradient_accumulation_steps` in config
- ✅ DeepSpeed handles everything automatically

---

### Step 6: Update Checkpointing

**Before (DDP)**:
```python
# Save checkpoint
if rank == 0:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),  # Note: .module
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, 'checkpoint.pt')

# Load checkpoint
checkpoint = torch.load('checkpoint.pt', map_location=f'cuda:{local_rank}')
model.module.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
```

**After (DeepSpeed)**:
```python
# Save checkpoint (all ranks participate)
client_state = {'epoch': epoch, 'loss': loss}
model_engine.save_checkpoint(
    save_dir='checkpoints',
    tag=f'epoch_{epoch}',
    client_state=client_state
)

# Load checkpoint (all ranks participate)
_, client_state = model_engine.load_checkpoint(
    load_dir='checkpoints',
    tag=f'epoch_{epoch}'
)
epoch = client_state['epoch']
loss = client_state['loss']
```

**Key Changes**:
- ❌ Don't use `if rank == 0` → all ranks participate
- ❌ Don't access `.module` → not needed
- ❌ Don't manually save optimizer/scheduler → handled by DeepSpeed
- ✅ Use `model_engine.save_checkpoint()` and `load_checkpoint()`
- ⚠️ DeepSpeed creates directory structure with multiple files

**Checkpoint Structure**:
```
checkpoints/
├── epoch_1/
│   ├── mp_rank_00_model_states.pt
│   ├── zero_pp_rank_0_mp_rank_00_optim_states.pt
│   ├── zero_pp_rank_1_mp_rank_00_optim_states.pt
│   └── ... (one per GPU for optimizer states)
```

---

## Side-by-Side Code Comparison

### Complete Example: Training Script

<table>
<tr>
<th>PyTorch DDP</th>
<th>DeepSpeed</th>
</tr>

<tr>
<td>

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DDP
from torch.cuda.amp import autocast, GradScaler

def main():
    # Initialize distributed
    dist.init_process_group('nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    # Model
    model = MyModel().cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4
    )

    # Mixed precision
    scaler = GradScaler()

    # Training loop
    model.train()
    for batch in dataloader:
        batch = {k: v.cuda() for k, v in batch.items()}

        optimizer.zero_grad()

        with autocast():
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )
        scaler.step(optimizer)
        scaler.update()

    # Save
    if dist.get_rank() == 0:
        torch.save({
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict()
        }, 'checkpoint.pt')

if __name__ == '__main__':
    main()
```

</td>
<td>

```python
import torch
import deepspeed

def main():
    # No manual init needed!

    # Model
    model = MyModel()

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = \
        deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config_params='ds_config.json'
        )

    # Training loop
    model_engine.train()
    for batch in dataloader:
        batch = {k: v.to(model_engine.device)
                for k, v in batch.items()}

        outputs = model_engine(**batch)
        loss = outputs.loss

        model_engine.backward(loss)
        model_engine.step()

    # Save
    model_engine.save_checkpoint(
        save_dir='checkpoints',
        tag='final'
    )

if __name__ == '__main__':
    main()
```

**ds_config.json**:
```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 4,
  "optimizer": {
    "type": "AdamW",
    "params": {"lr": 1e-4}
  },
  "fp16": {"enabled": true},
  "gradient_clipping": 1.0,
  "zero_optimization": {"stage": 2}
}
```

</td>
</tr>
</table>

**Lines of Code**:
- DDP: ~50 lines
- DeepSpeed: ~25 lines + 10 lines config = ~35 total
- **Reduction: 30% fewer lines**

---

## Configuration Mapping

### Batch Size

| DDP | DeepSpeed Config |
|-----|------------------|
| `batch_size = 32` in DataLoader | `"train_micro_batch_size_per_gpu": 32` |
| Manual gradient accumulation | `"gradient_accumulation_steps": 4` |
| N/A | `"train_batch_size": 128` (total effective) |

### Optimizer

| DDP | DeepSpeed Config |
|-----|------------------|
| `torch.optim.Adam(model.parameters(), lr=1e-4)` | `"optimizer": {"type": "Adam", "params": {"lr": 1e-4}}` |
| `torch.optim.AdamW(...)` | `"optimizer": {"type": "AdamW", ...}` |
| Custom optimizer | Pass to `deepspeed.initialize(optimizer=...)` |

### Mixed Precision

| DDP (PyTorch AMP) | DeepSpeed Config |
|-------------------|------------------|
| `autocast()` | `"fp16": {"enabled": true}` |
| `GradScaler()` | `"fp16": {"loss_scale": 0}` (dynamic) |
| N/A (not easily available) | `"bf16": {"enabled": true}` |

### Gradient Clipping

| DDP | DeepSpeed Config |
|-----|------------------|
| `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` | `"gradient_clipping": 1.0` |

### Learning Rate Schedule

| DDP | DeepSpeed Config |
|-----|------------------|
| `CosineAnnealingLR(optimizer, ...)` | `"scheduler": {"type": "WarmupLR", ...}` |
| Custom scheduler | Pass to `deepspeed.initialize(lr_scheduler=...)` |

---

## Performance Optimization

### Optimization 1: Enable ZeRO-2 for Memory Savings

```json
{
  "zero_optimization": {
    "stage": 2,  // Partition optimizer + gradients
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8
  }
}
```

**Expected Results**:
- Memory: 50-60% reduction vs DDP
- Speed: 5-10% slower than DDP

### Optimization 2: Enable FusedAdam

```json
{
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "zero_optimization": {
    "stage": 2
  }
}
```

DeepSpeed automatically uses `FusedAdam` which is **1.5-2× faster** than PyTorch's Adam.

### Optimization 3: Overlap Communication

```json
{
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,  // Overlap gradient comm with backward
    "contiguous_gradients": true  // Reduce memory fragmentation
  }
}
```

**Expected Results**:
- Speed: 10-15% faster than naive ZeRO-2

### Optimization 4: CPU Offloading (If Memory Constrained)

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

**Expected Results**:
- Memory: 70-80% reduction vs DDP
- Speed: 20-30% slower than ZeRO-3 without offload
- **Trade-off**: Enables training 2-3× larger models

---

## Common Migration Issues

### Issue 1: `AttributeError: 'DeepSpeedEngine' has no attribute 'module'`

**Problem**: Accessing `model.module` (DDP pattern)

**Solution**: DeepSpeed engine doesn't need `.module`

```python
# DDP
predictions = model.module.generate(...)

# DeepSpeed - Option 1: Use engine directly
predictions = model_engine.generate(...)

# DeepSpeed - Option 2: Access underlying module
predictions = model_engine.module.generate(...)
```

### Issue 2: Training slower after migration

**Problem**: Using default ZeRO-3 with small models

**Solution**: Use ZeRO-2 or ZeRO-1 for smaller models

```json
{
  "zero_optimization": {
    "stage": 2  // Change from 3 to 2
  }
}
```

**When to use each stage**:
- ZeRO-0: Models < 1B params (disable ZeRO)
- ZeRO-1: Models 1B-3B params
- ZeRO-2: Models 3B-13B params
- ZeRO-3: Models > 13B params

### Issue 3: Checkpoint loading fails

**Problem**: Trying to load DDP checkpoint into DeepSpeed

**Solution**: Convert checkpoint format

```python
# Load old DDP checkpoint
ddp_checkpoint = torch.load('ddp_checkpoint.pt')

# Create model
model = MyModel()

# Load weights
model.load_state_dict(ddp_checkpoint['model_state_dict'])

# Initialize DeepSpeed
model_engine, _, _, _ = deepspeed.initialize(
    model=model,  // Already has weights loaded
    model_parameters=model.parameters(),
    config_params='ds_config.json'
)

# Save as DeepSpeed checkpoint
model_engine.save_checkpoint('checkpoints', tag='converted')
```

### Issue 4: Batch size mismatch errors

**Problem**: DeepSpeed computes global batch size differently

**Solution**: Understand the formula

```
train_batch_size = micro_batch_size × grad_accum × num_gpus
```

**Example with 8 GPUs**:
```json
{
  "train_batch_size": 256,  // Global effective batch size
  "train_micro_batch_size_per_gpu": 4,  // Per GPU per step
  "gradient_accumulation_steps": 8  // 256 = 4 × 8 × 8
}
```

---

## Validation and Testing

### Step 1: Verify Correctness

Run both versions and compare:

```python
# DDP
ddp_losses = []
for batch in dataloader:
    loss = ddp_model(**batch).loss
    ddp_losses.append(loss.item())

# DeepSpeed
ds_losses = []
for batch in dataloader:
    loss = ds_model(**batch).loss
    ds_losses.append(loss.item())

# Compare (should be very close)
import numpy as np
print(f"DDP mean loss: {np.mean(ddp_losses):.4f}")
print(f"DS mean loss: {np.mean(ds_losses):.4f}")
print(f"Difference: {abs(np.mean(ddp_losses) - np.mean(ds_losses)):.6f}")
```

### Step 2: Benchmark Performance

```python
import time

# DDP
start = time.time()
for _ in range(100):
    loss = ddp_model(**batch).loss
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()
ddp_time = time.time() - start

# DeepSpeed
start = time.time()
for _ in range(100):
    loss = ds_model(**batch).loss
    ds_model.backward(loss)
    ds_model.step()
torch.cuda.synchronize()
ds_time = time.time() - start

print(f"DDP: {ddp_time:.2f}s")
print(f"DeepSpeed: {ds_time:.2f}s")
print(f"Speedup: {ddp_time/ds_time:.2f}×")
```

### Step 3: Memory Comparison

```bash
# Run DDP
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > ddp_mem.txt

# Run DeepSpeed
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > ds_mem.txt

# Compare
paste ddp_mem.txt ds_mem.txt | awk '{print "GPU mem: DDP="$1"MB, DS="$2"MB, Reduction="($1-$2)/$1*100"%"}'
```

---

## Migration Checklist

Use this checklist to track your migration progress:

- [ ] **Remove DDP imports and initialization**
  - [ ] Remove `torch.distributed.init_process_group()`
  - [ ] Remove `DDP()` wrapper
  - [ ] Remove manual device placement (`.cuda()`)

- [ ] **Add DeepSpeed initialization**
  - [ ] Add `import deepspeed`
  - [ ] Add `deepspeed.initialize()` call
  - [ ] Create `ds_config.json`

- [ ] **Update training loop**
  - [ ] Replace `optimizer.zero_grad()` with `model_engine.step()`
  - [ ] Replace `loss.backward()` with `model_engine.backward()`
  - [ ] Replace `optimizer.step()` with `model_engine.step()`
  - [ ] Use `model_engine.device` instead of `local_rank`

- [ ] **Configure features in ds_config.json**
  - [ ] Optimizer configuration
  - [ ] Mixed precision (FP16/BF16)
  - [ ] Gradient clipping
  - [ ] ZeRO stage
  - [ ] Batch sizes

- [ ] **Update checkpointing**
  - [ ] Replace `torch.save()` with `model_engine.save_checkpoint()`
  - [ ] Replace `torch.load()` with `model_engine.load_checkpoint()`
  - [ ] Remove `if rank == 0` guards

- [ ] **Update launch command**
  - [ ] Replace `torchrun` with `deepspeed`
  - [ ] Update command-line arguments

- [ ] **Test and validate**
  - [ ] Verify training loss matches DDP
  - [ ] Benchmark performance
  - [ ] Check memory usage
  - [ ] Test checkpoint save/load

---

## Additional Resources

- **[DeepSpeed Documentation](https://www.deepspeed.ai/)** - Official docs
- **[ZeRO-3 Concept to Code](../guides/ZeRO3_Concept_to_Code.md)** - Deep dive into ZeRO
- **[Troubleshooting Guide](../guides/Troubleshooting_Guide.md)** - Common issues
- **[Performance Benchmarks](../benchmarks/README.md)** - ZeRO stage comparison

---

## Next Steps

After successfully migrating to DeepSpeed:

1. **Optimize ZeRO stage**: Run benchmarks to find optimal stage for your model
2. **Enable offloading**: If still running out of memory, try CPU/NVMe offload
3. **Tune performance**: Experiment with `overlap_comm`, bucket sizes
4. **Scale up**: Train larger models or increase batch size

**Happy training with DeepSpeed!** 🚀
