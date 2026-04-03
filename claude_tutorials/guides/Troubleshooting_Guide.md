# DeepSpeed Troubleshooting Guide

A comprehensive guide to diagnosing and fixing common DeepSpeed issues. Organized by error type with clear solutions and prevention strategies.

---

## Table of Contents

1. [Out of Memory (OOM) Errors](#out-of-memory-oom-errors)
2. [Initialization and Setup Errors](#initialization-and-setup-errors)
3. [Communication and NCCL Errors](#communication-and-nccl-errors)
4. [Training Instability and NaN Loss](#training-instability-and-nan-loss)
5. [Checkpoint and Saving Errors](#checkpoint-and-saving-errors)
6. [Performance Issues](#performance-issues)
7. [Configuration Errors](#configuration-errors)
8. [Multi-Node Training Issues](#multi-node-training-issues)
9. [Offloading Issues](#offloading-issues)
10. [Mixed Precision and Overflow](#mixed-precision-and-overflow)

---

## Out of Memory (OOM) Errors

### Error 1: `CUDA out of memory` during initialization

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
(GPU 0; 79.20 GiB total capacity; 76.50 GiB already allocated;
1.50 GiB free; 77.00 GiB reserved in total by PyTorch)
```

**Causes**:
- Model too large for GPU memory
- Batch size too large
- Sequence length too long
- Incorrect ZeRO stage for model size

**Solutions**:

#### Solution A: Increase ZeRO Stage
```json
{
  "zero_optimization": {
    "stage": 3  // Increase from 0/1/2 to 3
  }
}
```

#### Solution B: Reduce Batch Size
```json
{
  "train_batch_size": 16,  // Reduce this
  "train_micro_batch_size_per_gpu": 1,  // Or this
  "gradient_accumulation_steps": 16  // Increase to maintain effective batch size
}
```

#### Solution C: Enable Activation Checkpointing
```python
# In your model
model.gradient_checkpointing_enable()

# Or in config
{
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true,
    "contiguous_memory_optimization": true,
    "number_checkpoints": 4
  }
}
```

#### Solution D: Enable CPU/NVMe Offloading
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

**Memory Reduction Table**:
| Change | Memory Saved | Speed Impact |
|--------|--------------|--------------|
| ZeRO-1 → ZeRO-2 | ~2× optimizer+grad | 5-10% slower |
| ZeRO-2 → ZeRO-3 | ~linear scaling | 15-25% slower |
| Activation checkpoint | 40-60% | 20-33% slower |
| CPU offload (optimizer) | 40-50% | 10-20% slower |
| CPU offload (full) | 70-80% | 30-50% slower |

---

### Error 2: OOM during forward pass

**Symptoms**:
```
RuntimeError: CUDA out of memory during forward()
Allocated: 75.2 GB, Reserved: 78.5 GB
```

**Causes**:
- Activations too large (long sequences or large hidden dimensions)
- Too many active parameters with ZeRO-3

**Solutions**:

#### Solution A: Reduce Prefetch Parameters (ZeRO-3)
```json
{
  "zero_optimization": {
    "stage": 3,
    "stage3_max_live_parameters": 1e8,  // Reduce from 1e9
    "stage3_max_reuse_distance": 1e8,   // Reduce from 1e9
    "stage3_prefetch_bucket_size": 5e7  // Reduce from auto
  }
}
```

#### Solution B: Partition Activations
```json
{
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true,
    "contiguous_memory_optimization": true,
    "number_checkpoints": null,  // Auto-compute
    "synchronize_checkpoint_boundary": false,
    "profile": false
  }
}
```

#### Solution C: Reduce Sequence Length
```python
# In your data loader
max_length = 512  # Reduce from 1024 or 2048

# Or use dynamic padding
tokenizer(text, truncation=True, max_length=512, padding='max_length')
```

---

### Error 3: OOM during backward pass

**Symptoms**:
```
RuntimeError: CUDA out of memory during loss.backward()
```

**Causes**:
- Gradients accumulating in memory
- Not releasing intermediate activations

**Solutions**:

#### Solution A: Enable Contiguous Gradients
```json
{
  "zero_optimization": {
    "stage": 2,  // or 3
    "contiguous_gradients": true,  // Reduces fragmentation
    "overlap_comm": true  // Overlaps gradient comm
  }
}
```

#### Solution B: Clear Cache Between Steps
```python
def training_step(batch):
    outputs = model(**batch)
    loss = outputs.loss
    model.backward(loss)
    model.step()

    # Clear cache periodically
    if step % 10 == 0:
        torch.cuda.empty_cache()
```

---

## Initialization and Setup Errors

### Error 4: `deepspeed.initialize() failed`

**Symptoms**:
```
TypeError: initialize() got an unexpected keyword argument 'config'
ValueError: config file does not exist
```

**Causes**:
- Incorrect DeepSpeed API usage
- Missing or invalid config file
- Wrong DeepSpeed version

**Solutions**:

#### Solution A: Correct API Usage
```python
# INCORRECT
model_engine = deepspeed.initialize(
    model=model,
    config="ds_config.json"  # Wrong parameter name
)

# CORRECT
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config_params="ds_config.json"  # or config=dict
)
```

#### Solution B: Verify Config File Exists
```python
import os
config_path = "ds_config.json"
assert os.path.exists(config_path), f"Config not found: {config_path}"
```

#### Solution C: Check DeepSpeed Version
```bash
# Check version
pip show deepspeed

# Upgrade to latest
pip install --upgrade deepspeed

# Or install specific version
pip install deepspeed==0.12.0
```

---

### Error 5: `Rank 0 initialized but other ranks stuck`

**Symptoms**:
```
[Rank 0] DeepSpeed initialized successfully
[Rank 1] (hangs indefinitely)
[Rank 2] (hangs indefinitely)
```

**Causes**:
- Inconsistent config across ranks
- File system race condition
- Network/NCCL initialization issues

**Solutions**:

#### Solution A: Use Barrier After Config Creation
```python
# On rank 0: Create config
if local_rank == 0:
    with open('ds_config.json', 'w') as f:
        json.dump(config, f)

# Barrier to ensure all ranks see the file
torch.distributed.barrier()

# All ranks: Load config
with open('ds_config.json', 'r') as f:
    config = json.load(f)

model_engine, _, _, _ = deepspeed.initialize(...)
```

#### Solution B: Pass Config as Dict (Not File)
```python
# Create config as dict
config = {
    "train_batch_size": 32,
    "zero_optimization": {"stage": 3}
}

# Pass dict directly (no file I/O)
model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=config  # Dict, not file path
)
```

#### Solution C: Check NCCL Environment
```bash
# Set NCCL debug level
export NCCL_DEBUG=INFO

# Run training
deepspeed train.py

# Look for NCCL initialization errors in output
```

---

## Communication and NCCL Errors

### Error 6: `NCCL operation timed out`

**Symptoms**:
```
RuntimeError: NCCL error in: /pytorch/torch/lib/c10d/ProcessGroupNCCL.cpp:825
unhandled system error, NCCL version 2.10.3
Last error: Timed out
```

**Causes**:
- Network connectivity issues
- Firewall blocking NCCL ports
- Slow collective operations (large gradients)
- Rank mismatch or crash

**Solutions**:

#### Solution A: Increase NCCL Timeout
```bash
# Default is 10 minutes, increase to 30
export NCCL_TIMEOUT=1800

# Or disable timeout (for debugging only)
export NCCL_TIMEOUT=0
```

#### Solution B: Check Network Connectivity
```bash
# Test network between nodes
# On node 0:
iperf3 -s

# On node 1:
iperf3 -c <node0_ip>

# Should see > 10 Gbps for InfiniBand, > 1 Gbps for Ethernet
```

#### Solution C: Verify NCCL Configuration
```bash
# Use optimal NCCL settings
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_NET_GDR_LEVEL=3  # Enable GPU Direct RDMA
export NCCL_SOCKET_IFNAME=ib0  # Specify network interface

# For Ethernet (not IB)
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
```

#### Solution D: Reduce Communication Volume
```json
{
  "zero_optimization": {
    "stage": 3,
    "reduce_bucket_size": 5e7,  // Reduce from default
    "allgather_bucket_size": 5e7  // Reduce from default
  }
}
```

---

### Error 7: `NCCL all-reduce failed`

**Symptoms**:
```
RuntimeError: NCCL error: unhandled system error
Segmentation fault (core dumped)
```

**Causes**:
- GPU memory corruption
- Incompatible NCCL version
- Driver/CUDA version mismatch

**Solutions**:

#### Solution A: Check CUDA/Driver Compatibility
```bash
# Check CUDA version
nvcc --version

# Check driver version
nvidia-smi

# Verify compatibility
# CUDA 11.8 requires driver >= 450.80.02
# CUDA 12.0 requires driver >= 525.60.13
```

#### Solution B: Rebuild NCCL
```bash
# Uninstall existing NCCL
pip uninstall nccl

# Reinstall DeepSpeed with NCCL rebuild
DS_BUILD_OPS=1 pip install deepspeed --global-option="build_ext"
```

#### Solution C: Use Compatible PyTorch + NCCL
```bash
# Install PyTorch with bundled NCCL
pip install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

---

## Training Instability and NaN Loss

### Error 8: Loss becomes NaN

**Symptoms**:
```
Step 100: loss = 2.453
Step 101: loss = 2.398
Step 102: loss = nan
```

**Causes**:
- FP16 overflow
- Learning rate too high
- Gradient explosion
- Incorrect loss scaling

**Solutions**:

#### Solution A: Enable Dynamic Loss Scaling
```json
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,  // 0 = dynamic
    "initial_scale_power": 16,  // Start at 2^16
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  }
}
```

#### Solution B: Use BF16 Instead of FP16
```json
{
  "bf16": {
    "enabled": true  // More stable than FP16
  },
  "fp16": {
    "enabled": false  // Disable FP16
  }
}
```

#### Solution C: Gradient Clipping
```json
{
  "gradient_clipping": 1.0  // Clip gradients to max norm
}
```

#### Solution D: Reduce Learning Rate
```python
# Reduce initial LR
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # Instead of 1e-4

# Or use warmup
{
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 1e-4,
      "warmup_num_steps": 1000
    }
  }
}
```

#### Solution E: Debug Gradients
```python
# Add gradient checking
def check_gradients(model, step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"Step {step}: NaN/Inf gradient in {name}")
                print(f"  Param norm: {param.norm()}")
                print(f"  Grad norm: {grad_norm}")
                return True
    return False

# In training loop
loss = model(**batch).loss
model.backward(loss)

if check_gradients(model, step):
    print("Skipping step due to bad gradients")
    model.zero_grad()
else:
    model.step()
```

---

### Error 9: Training hangs after some steps

**Symptoms**:
```
Step 1000: loss = 2.453, time = 0.5s
Step 1001: loss = 2.398, time = 0.5s
Step 1002: (hangs indefinitely)
```

**Causes**:
- Deadlock in collective operations
- Uneven data distribution causing some GPUs to finish early
- Checkpoint saving hanging

**Solutions**:

#### Solution A: Add Timeouts to DataLoader
```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    timeout=60,  // Add timeout
    pin_memory=True
)
```

#### Solution B: Synchronize Before Checkpoints
```python
def save_checkpoint(model_engine, step):
    # Ensure all ranks reach this point
    torch.distributed.barrier()

    # Save checkpoint
    model_engine.save_checkpoint(
        save_dir='checkpoints',
        tag=f'step_{step}'
    )

    # Ensure all ranks finish saving
    torch.distributed.barrier()
```

#### Solution C: Use Drop Last in DataLoader
```python
# Ensure all GPUs process same number of batches
dataloader = DataLoader(
    dataset,
    batch_size=32,
    drop_last=True  // Important for distributed training
)
```

---

## Checkpoint and Saving Errors

### Error 10: `Failed to save checkpoint`

**Symptoms**:
```
OSError: [Errno 28] No space left on device
RuntimeError: Unable to save checkpoint at step 1000
```

**Causes**:
- Disk full
- Permission issues
- Network file system issues
- ZeRO-3 state dict too large

**Solutions**:

#### Solution A: Check Disk Space
```bash
# Check available space
df -h /path/to/checkpoints

# Clean old checkpoints
rm -rf checkpoints/old_checkpoint_*
```

#### Solution B: Save ZeRO Checkpoint (Not Full State Dict)
```python
# INCORRECT: Tries to gather all params
torch.save(model.state_dict(), 'model.pt')

# CORRECT: Saves ZeRO checkpoint
model_engine.save_checkpoint(
    save_dir='checkpoints',
    tag='step_1000',
    client_state={'step': 1000}
)
```

#### Solution C: Configure Checkpoint Saving
```json
{
  "checkpoint": {
    "tag_validation": "Strict",
    "load_universal": false,
    "use_node_local_storage": false
  }
}
```

#### Solution D: Save Only on Rank 0
```python
if torch.distributed.get_rank() == 0:
    # Save lightweight metadata
    torch.save({
        'step': step,
        'config': config,
        'metrics': metrics
    }, 'metadata.pt')

# Let DeepSpeed handle model checkpointing
model_engine.save_checkpoint('checkpoints', tag=f'step_{step}')
```

---

### Error 11: `Cannot load checkpoint`

**Symptoms**:
```
FileNotFoundError: Checkpoint directory not found
RuntimeError: Checkpoint mismatch: expected ZeRO stage 3, got stage 2
```

**Causes**:
- Checkpoint doesn't exist
- ZeRO stage mismatch between save and load
- Wrong checkpoint format

**Solutions**:

#### Solution A: Verify Checkpoint Structure
```bash
# ZeRO checkpoint structure
checkpoints/
├── step_1000/
│   ├── mp_rank_00_model_states.pt
│   ├── zero_pp_rank_0_mp_rank_00_optim_states.pt
│   └── latest  # Symlink or tag file
```

#### Solution B: Load with Matching Configuration
```python
# Use SAME ZeRO stage when loading
config = {
    "zero_optimization": {
        "stage": 3  // Must match checkpoint stage
    }
}

model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    config=config
)

# Load checkpoint
_, client_state = model_engine.load_checkpoint(
    load_dir='checkpoints',
    tag='step_1000'
)
```

#### Solution C: Convert Checkpoint Format
```python
# Convert ZeRO checkpoint to universal format
from deepspeed.checkpoint import DeepSpeedCheckpoint

ds_checkpoint = DeepSpeedCheckpoint('checkpoints/step_1000')
state_dict = ds_checkpoint.get_zero_checkpoint_state_dict()

# Save as standard PyTorch checkpoint
torch.save(state_dict, 'model.pt')
```

---

## Performance Issues

### Error 12: Training is very slow

**Symptoms**:
- 10× slower than expected
- Low GPU utilization (< 50%)
- High CPU usage

**Causes**:
- CPU bottleneck in data loading
- Too many gradient accumulation steps
- Excessive logging or checkpointing
- Suboptimal communication

**Solutions**:

#### Solution A: Optimize Data Loading
```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,  # Increase workers
    pin_memory=True,  # Enable for GPU transfer
    prefetch_factor=2,  # Prefetch batches
    persistent_workers=True  # Keep workers alive
)
```

#### Solution B: Reduce Logging Frequency
```json
{
  "steps_per_print": 100,  // Reduce from 10
  "wall_clock_breakdown": false,  // Disable unless debugging
  "dump_state": false  // Disable unless debugging
}
```

#### Solution C: Optimize Communication
```json
{
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,  // Overlap communication with computation
    "contiguous_gradients": true,  // Reduce fragmentation
    "reduce_bucket_size": 5e8,  // Tune bucket size
    "allgather_bucket_size": 5e8,
    "reduce_scatter": true  // Enable reduce-scatter optimization
  }
}
```

#### Solution D: Profile the Training
```bash
# Enable profiling
export DEEPSPEED_PROFILE=1

# Run training
deepspeed train.py

# Check profile output
cat deepspeed_profile.json
```

---

### Error 13: High memory usage with low GPU utilization

**Symptoms**:
- GPU memory 90%+ full
- GPU utilization < 30%
- Training very slow

**Causes**:
- Batch size too small for GPU
- Too much CPU offloading
- Activation checkpointing overhead

**Solutions**:

#### Solution A: Increase Batch Size with Gradient Accumulation
```json
{
  "train_batch_size": 128,  // Effective batch size
  "train_micro_batch_size_per_gpu": 8,  // Increase from 1
  "gradient_accumulation_steps": 16  // Reduce from 128
}
```

#### Solution B: Reduce CPU Offloading
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true,
      "fast_init": false
    },
    // Remove parameter offloading if not needed
    // "offload_param": { ... }
  }
}
```

#### Solution C: Optimize Activation Checkpointing
```python
# Use selective activation checkpointing
model.gradient_checkpointing_enable()

# Or checkpoint every N layers
for i, layer in enumerate(model.layers):
    if i % 3 == 0:  # Checkpoint every 3rd layer
        layer = torch.utils.checkpoint(layer)
```

---

## Configuration Errors

### Error 14: `Config parameter not recognized`

**Symptoms**:
```
Warning: Unused configuration key: zero_optimisation
DeepSpeedConfigError: Unknown parameter 'optimiser'
```

**Causes**:
- Typo in config parameter name
- Deprecated parameter
- Wrong parameter location in config

**Solutions**:

#### Solution A: Common Typos
```json
// INCORRECT (British spelling)
{
  "zero_optimisation": { ... },  // ❌
  "optimiser": { ... }  // ❌
}

// CORRECT (American spelling)
{
  "zero_optimization": { ... },  // ✅
  "optimizer": { ... }  // ✅
}
```

#### Solution B: Validate Config
```python
import deepspeed
from deepspeed.runtime.config import DeepSpeedConfig

# Validate config before using
config = {...}
ds_config = DeepSpeedConfig(config)

# Check for warnings
if ds_config.monitor_config.enabled:
    print("Monitoring enabled")
```

#### Solution C: Use Latest Config Schema
```bash
# Get example config
deepspeed --help-all > deepspeed_options.txt

# Or check official docs
https://www.deepspeed.ai/docs/config-json/
```

---

### Error 15: `Batch size mismatch`

**Symptoms**:
```
AssertionError: train_batch_size (64) must equal
train_micro_batch_size_per_gpu (4) * gradient_accumulation_steps (8) * num_gpus (4)
```

**Causes**:
- Inconsistent batch size configuration
- Not accounting for number of GPUs

**Solutions**:

#### Solution A: Correct Batch Size Math
```json
// Formula: train_batch_size = micro_batch * grad_accum * num_gpus
{
  "train_batch_size": 128,  // Total effective batch size
  "train_micro_batch_size_per_gpu": 4,  // Per GPU per step
  "gradient_accumulation_steps": 8  // Accumulation steps
}
// With 4 GPUs: 4 * 8 * 4 = 128 ✅
```

#### Solution B: Use "auto" for train_batch_size
```json
{
  "train_batch_size": "auto",  // DeepSpeed computes automatically
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 8
}
```

---

## Multi-Node Training Issues

### Error 16: Multi-node training won't start

**Symptoms**:
```
[Node 0] Waiting for other nodes...
[Node 1] Connection refused to node 0
```

**Causes**:
- Firewall blocking ports
- Wrong master address/port
- SSH keys not configured

**Solutions**:

#### Solution A: Open Required Ports
```bash
# Open ports 29500-29600 (PyTorch distributed default range)
sudo ufw allow 29500:29600/tcp
sudo ufw allow 29500:29600/udp

# For NCCL (if using IB)
sudo ufw allow 50000:51000/tcp
```

#### Solution B: Verify SSH Configuration
```bash
# On master node, test SSH to all workers
ssh worker1 'echo Success'
ssh worker2 'echo Success'

# Setup passwordless SSH if needed
ssh-keygen -t rsa
ssh-copy-id worker1
ssh-copy-id worker2
```

#### Solution C: Launch with Correct Hostfile
```bash
# Create hostfile
cat > hostfile <<EOF
master slots=8
worker1 slots=8
worker2 slots=8
EOF

# Launch with explicit master address
deepspeed --hostfile=hostfile \
    --master_addr=master \
    --master_port=29500 \
    train.py --deepspeed_config=ds_config.json
```

---

### Error 17: Nodes have different speeds

**Symptoms**:
- Node 0: 0.5s/step
- Node 1: 0.8s/step
- Training as slow as slowest node

**Causes**:
- Hardware differences between nodes
- Network bottleneck
- Different data loading speeds

**Solutions**:

#### Solution A: Profile Each Node
```bash
# On each node separately
CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmark.py

# Compare results
Node 0: 2000 tokens/sec
Node 1: 1500 tokens/sec  # Slower!
```

#### Solution B: Balance Data Loading
```python
# Use same random seed on all nodes
torch.manual_seed(42 + rank)

# Ensure same number of batches
dataloader = DataLoader(
    dataset,
    batch_size=32,
    drop_last=True,  # Critical for multi-node
    shuffle=True
)
```

#### Solution C: Check Network Bandwidth
```bash
# Test inter-node bandwidth
# On node 0:
iperf3 -s

# On node 1:
iperf3 -c node0 -P 8  # 8 parallel streams

# Should see > 10 Gbps for good performance
```

---

## Offloading Issues

### Error 18: CPU offload slower than expected

**Symptoms**:
- Training 5-10× slower than expected
- High CPU usage but low GPU usage

**Causes**:
- Slow CPU-GPU transfers
- Not using pinned memory
- CPU can't keep up with optimizer updates

**Solutions**:

#### Solution A: Enable Pinned Memory
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true,  // Critical for speed
      "fast_init": false
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true  // Critical for speed
    }
  }
}
```

#### Solution B: Use DeepSpeedCPUAdam
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  },
  "optimizer": {
    "type": "AdamW",  // DeepSpeed will use CPU version
    "params": {
      "lr": 1e-4
    }
  }
}
```

#### Solution C: Tune Overlap and Prefetch
```json
{
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "prefetch_bucket_size": 5e7,
    "max_reuse_distance": 1e9,
    "sub_group_size": 1e9
  }
}
```

---

### Error 19: NVMe offload fails

**Symptoms**:
```
OSError: NVMe path not accessible: /mnt/nvme
RuntimeError: AIO not available
```

**Causes**:
- NVMe path doesn't exist
- No write permissions
- AIO library not installed

**Solutions**:

#### Solution A: Setup NVMe Path
```bash
# Create NVMe directory
sudo mkdir -p /mnt/nvme

# Set permissions
sudo chown $USER:$USER /mnt/nvme
chmod 755 /mnt/nvme

# Verify writable
touch /mnt/nvme/test.txt
rm /mnt/nvme/test.txt
```

#### Solution B: Install AIO
```bash
# Install libaio
sudo apt-get install libaio-dev

# Rebuild DeepSpeed with AIO
DS_BUILD_AIO=1 pip install deepspeed --force-reinstall --no-cache-dir
```

#### Solution C: Verify NVMe Speed
```bash
# Test write speed
dd if=/dev/zero of=/mnt/nvme/test.img bs=1G count=1 oflag=direct

# Should see > 3 GB/s for good NVMe
# If < 500 MB/s, may be regular SSD, not NVMe
```

#### Solution D: Configure AIO Parameters
```json
{
  "aio": {
    "block_size": 1048576,  // 1 MB
    "queue_depth": 8,
    "thread_count": 1,
    "single_submit": false,
    "overlap_events": true
  }
}
```

---

## Mixed Precision and Overflow

### Error 20: FP16 overflow

**Symptoms**:
```
Step 50: loss = 2.453, scale = 65536
Step 51: loss = nan, scale = 32768
Step 52: loss = nan, scale = 16384
...
Step 60: loss = nan, scale = 1  // Scale decreased to 1
```

**Causes**:
- Gradients too large for FP16 range
- Loss scale keeps decreasing
- Model weights exploding

**Solutions**:

#### Solution A: Switch to BF16
```json
{
  "bf16": {
    "enabled": true  // Better dynamic range than FP16
  },
  "fp16": {
    "enabled": false
  }
}
```

#### Solution B: Tune Loss Scaling
```json
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,  // Dynamic scaling
    "initial_scale_power": 20,  // Start higher (2^20)
    "loss_scale_window": 500,  // Increase window
    "min_loss_scale": 128  // Don't go below 128
  }
}
```

#### Solution C: Gradient Clipping
```json
{
  "gradient_clipping": 1.0  // Clip before scaling
}
```

---

## Debugging Tools

### Enable Detailed Logging

```bash
# DeepSpeed debug output
export DEEPSPEED_DEBUG=1

# NCCL debug output
export NCCL_DEBUG=INFO

# PyTorch distributed debug
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Run training
deepspeed train.py 2>&1 | tee debug.log
```

### Memory Profiling

```python
import torch

# At start of training
torch.cuda.reset_peak_memory_stats()

# After each step
allocated = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.memory_reserved() / 1e9
peak = torch.cuda.max_memory_allocated() / 1e9

print(f"Step {step}: Allocated {allocated:.2f}GB, "
      f"Reserved {reserved:.2f}GB, Peak {peak:.2f}GB")
```

### Gradient Debugging

```python
def debug_gradients(model, step):
    """Print gradient statistics."""
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if grad_norm > 100:
                print(f"[Step {step}] Large gradient in {name}: {grad_norm}")

    print(f"[Step {step}] Grad norm: mean={np.mean(grad_norms):.4f}, "
          f"max={np.max(grad_norms):.4f}, min={np.min(grad_norms):.4f}")
```

---

## Quick Reference: Error Code to Solution

| Error Pattern | First Thing to Try |
|---------------|-------------------|
| `CUDA out of memory` | Increase ZeRO stage or reduce batch size |
| `NCCL timeout` | Export NCCL_TIMEOUT=1800 |
| `Loss = NaN` | Enable BF16 or dynamic loss scaling |
| `Cannot load checkpoint` | Verify ZeRO stage matches |
| `Nodes hanging` | Check SSH and firewall |
| `Training slow` | Increase num_workers in DataLoader |
| `NVMe failed` | Install libaio and rebuild DeepSpeed |

---

## Additional Resources

- **[DeepSpeed Documentation](https://www.deepspeed.ai/)** - Official docs
- **[DeepSpeed GitHub Issues](https://github.com/microsoft/DeepSpeed/issues)** - Community solutions
- **[ZeRO-3 Concept to Code](./ZeRO3_Concept_to_Code.md)** - Understanding ZeRO internals
- **[Distributed Training Guide](./Distributed_Training_Guide.md)** - Complete data flow

---

## Getting Help

If you're still stuck after trying these solutions:

1. **Check DeepSpeed version**: `pip show deepspeed`
2. **Enable debug logging**: `export DEEPSPEED_DEBUG=1`
3. **Create minimal reproduction** with GPT-2 or small model
4. **Post issue** with full error log and config

**Template for bug reports**:
```
DeepSpeed version: X.Y.Z
PyTorch version: X.Y.Z
CUDA version: X.Y
Number of GPUs: X
GPU type: A100 / V100 / etc

Config:
{
  "zero_optimization": {"stage": 3}
}

Error:
RuntimeError: ...

Full log:
(attach complete error output)
```
