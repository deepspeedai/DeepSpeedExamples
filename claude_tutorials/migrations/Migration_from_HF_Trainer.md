# Migrating from HuggingFace Trainer (without DeepSpeed) to HuggingFace Trainer with DeepSpeed

A comprehensive guide for enabling DeepSpeed in your existing HuggingFace Trainer code with minimal changes.

---

## Table of Contents

1. [Why Add DeepSpeed to HF Trainer?](#why-add-deepspeed-to-hf-trainer)
2. [Quick Start (2 Minutes)](#quick-start-2-minutes)
3. [Detailed Integration Steps](#detailed-integration-steps)
4. [Configuration Guide](#configuration-guide)
5. [Advanced Features](#advanced-features)
6. [Performance Optimization](#performance-optimization)
7. [Common Issues](#common-issues)
8. [Best Practices](#best-practices)

---

## Why Add DeepSpeed to HF Trainer?

### The Good News

**HuggingFace Trainer already has built-in DeepSpeed support!** You don't need to rewrite your training code. You just need to:
1. Create a DeepSpeed config file
2. Pass it to `TrainingArguments`
3. Enjoy memory savings and speed improvements!

### Key Benefits

| Without DeepSpeed | With DeepSpeed (ZeRO-3) |
|-------------------|------------------------|
| Model must fit in single GPU | Can train models 10-100× larger |
| Limited batch size | Larger batches = faster training |
| Single-node scaling | Efficient multi-node training |
| Manual mixed precision | Automatic FP16/BF16 optimization |
| Standard optimizers | Optimized kernels (FusedAdam, CPUAdam) |

### When to Use

✅ **Use DeepSpeed if**:
- Running out of GPU memory
- Training models > 1B parameters
- Want to train faster
- Need to scale to multiple nodes
- Want automatic optimization

⚠️ **Skip DeepSpeed if**:
- Model comfortably fits in single GPU (< 500M params)
- Using very simple training scripts
- Need maximum simplicity

---

## Quick Start (2 Minutes)

### Step 1: Your Existing Code (No Changes Needed!)

```python
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM

# Your existing code - works as-is!
model = AutoModelForCausalLM.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    # ... other args
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

### Step 2: Create DeepSpeed Config

Create `ds_config.json`:

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "fp16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 2
  }
}
```

### Step 3: Enable DeepSpeed (One Line!)

```python
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    deepspeed="ds_config.json",  # ← ADD THIS ONE LINE!
    # ... other args (no changes needed)
)
```

### Step 4: Launch Training

```bash
# Before
python train.py

# After (DeepSpeed enabled)
deepspeed train.py

# Or with specific number of GPUs
deepspeed --num_gpus=8 train.py
```

**That's it!** Your existing HF Trainer code now uses DeepSpeed.

---

## Detailed Integration Steps

### Step 1: Understand "auto" Values

DeepSpeed config supports "auto" to inherit from `TrainingArguments`:

```json
{
  "train_batch_size": "auto",  // Inherits from per_device_train_batch_size × num_gpus
  "train_micro_batch_size_per_gpu": "auto",  // Inherits from per_device_train_batch_size
  "gradient_accumulation_steps": "auto",  // Inherits from TrainingArguments
  "fp16": {
    "enabled": "auto"  // Inherits from fp16=True in TrainingArguments
  },
  "bf16": {
    "enabled": "auto"  // Inherits from bf16=True in TrainingArguments
  }
}
```

**Recommendation**: Use "auto" for maximum compatibility with existing code.

---

### Step 2: Choose ZeRO Stage

Different stages for different needs:

#### ZeRO-0 (Disabled)
```json
{
  "zero_optimization": {
    "stage": 0
  }
}
```
- **Use for**: Models < 1B params
- **Memory savings**: None (same as vanilla HF Trainer)
- **Speed**: Baseline (no DeepSpeed overhead)

#### ZeRO-1 (Optimizer State Partitioning)
```json
{
  "zero_optimization": {
    "stage": 1
  }
}
```
- **Use for**: Models 1B-3B params
- **Memory savings**: 4× for optimizer states
- **Speed**: ~5% slower than ZeRO-0

#### ZeRO-2 (+ Gradient Partitioning) ⭐ **Recommended Starting Point**
```json
{
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "overlap_comm": true
  }
}
```
- **Use for**: Models 3B-13B params
- **Memory savings**: 8× (optimizer + gradients)
- **Speed**: ~10-15% slower than ZeRO-1
- **Sweet spot**: Best balance for most use cases

#### ZeRO-3 (+ Parameter Partitioning)
```json
{
  "zero_optimization": {
    "stage": 3,
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto"
  }
}
```
- **Use for**: Models > 13B params
- **Memory savings**: Linear scaling with GPUs
- **Speed**: ~20-30% slower than ZeRO-2
- **Enables**: Training models that don't fit in single GPU

---

### Step 3: Configure Mixed Precision

#### Option 1: Control from TrainingArguments (Recommended)

```python
training_args = TrainingArguments(
    output_dir="./output",
    fp16=True,  # Enable FP16
    # OR
    bf16=True,  # Enable BF16 (more stable, if supported)
    deepspeed="ds_config.json"
)
```

```json
{
  "fp16": {
    "enabled": "auto"  // Inherits from TrainingArguments
  }
}
```

#### Option 2: Control from DeepSpeed Config

```json
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,  // 0 = dynamic loss scaling
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  }
}
```

**Recommendation**: Use BF16 if your GPUs support it (A100, H100):
```python
training_args = TrainingArguments(
    bf16=True,  # More stable than FP16
    deepspeed="ds_config.json"
)
```

---

### Step 4: Configure Optimizer

#### Option 1: Let HF Trainer Create Optimizer (Default)

```python
# TrainingArguments
training_args = TrainingArguments(
    learning_rate=5e-5,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    weight_decay=0.01,
    deepspeed="ds_config.json"
)
```

```json
{
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",  // Inherits from TrainingArguments
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }
  }
}
```

#### Option 2: Configure in DeepSpeed Config

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
  }
}
```

#### Option 3: Use CPU-Offloaded Optimizer (For Large Models)

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
    "type": "AdamW",  // DeepSpeed will use CPUAdam
    "params": {
      "lr": 1e-4
    }
  }
}
```

**Recommendation**: Use "auto" for simplicity, unless you need CPU offloading.

---

### Step 5: Configure Gradient Accumulation

#### Control from TrainingArguments (Recommended)

```python
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # Accumulate over 8 steps
    deepspeed="ds_config.json"
)
```

```json
{
  "gradient_accumulation_steps": "auto"  // Inherits from TrainingArguments
}
```

#### Or Configure in DeepSpeed Config

```json
{
  "train_batch_size": 128,  // Total effective batch size
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 8  // 128 = 4 × 8 × num_gpus
}
```

---

## Configuration Guide

### Minimal Config (Recommended Starting Point)

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "fp16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 2
  }
}
```

**Why minimal?**
- Most values inherited from `TrainingArguments`
- Easy to maintain
- Works with existing code
- Provides good memory savings (ZeRO-2)

---

### Optimized Config for Speed

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "bf16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8,
    "reduce_scatter": true
  }
}
```

**Expected improvements**:
- 10-15% faster than basic ZeRO-2
- Better GPU utilization
- Lower memory fragmentation

---

### Optimized Config for Memory

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "fp16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 3,
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
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

**Expected improvements**:
- 10-20× memory reduction vs vanilla HF Trainer
- Can train models 10× larger
- 2-3× slower than ZeRO-2

---

### Config for Multi-Node Training

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "bf16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8
  },
  "steps_per_print": 100,
  "wall_clock_breakdown": false
}
```

Launch with hostfile:
```bash
deepspeed --hostfile=hostfile --master_port=29500 train.py
```

---

## Advanced Features

### Feature 1: Activation Checkpointing

Free up memory by recomputing activations during backward pass.

```python
# Enable in model
model.gradient_checkpointing_enable()

# Configure in TrainingArguments
training_args = TrainingArguments(
    gradient_checkpointing=True,
    deepspeed="ds_config.json"
)
```

```json
{
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": true,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
  }
}
```

**Memory savings**: 40-60% of activation memory
**Speed impact**: 20-33% slower (extra recomputation)

---

### Feature 2: CPU Offloading

Offload optimizer states and parameters to CPU RAM.

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true,
      "buffer_count": 4,
      "fast_init": false
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true,
      "buffer_count": 5,
      "buffer_size": 1e8,
      "max_in_cpu": 1e9
    }
  }
}
```

**When to use**:
- Model doesn't fit in GPU with ZeRO-3 alone
- Have lots of CPU RAM (4-8× GPU RAM)
- Can tolerate 20-40% slowdown

---

### Feature 3: NVMe Offloading

Offload parameters to NVMe SSD for extreme memory savings.

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "nvme",
      "nvme_path": "/local_nvme",
      "pin_memory": true,
      "buffer_count": 5,
      "buffer_size": 1e8
    }
  },
  "aio": {
    "block_size": 1048576,
    "queue_depth": 8,
    "thread_count": 1,
    "single_submit": false,
    "overlap_events": true
  }
}
```

**When to use**:
- Model doesn't fit in CPU RAM
- Have fast NVMe (PCIe 4.0+, > 5 GB/s)
- Training models 50B+ parameters

**Setup**:
```bash
# Install libaio
sudo apt-get install libaio-dev

# Rebuild DeepSpeed with AIO
DS_BUILD_AIO=1 pip install deepspeed --force-reinstall
```

---

### Feature 4: Zero-Infinity (ZeRO-3 + NVMe)

Train models with **infinite memory** using GPU + CPU + NVMe.

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "nvme",
      "nvme_path": "/local_nvme",
      "pin_memory": true,
      "buffer_count": 5,
      "buffer_size": 1e8
    }
  },
  "aio": {
    "block_size": 1048576,
    "queue_depth": 8,
    "thread_count": 1
  }
}
```

**Enables**: Training 1T+ parameter models on modest hardware

---

## Performance Optimization

### Tip 1: Use Larger Micro Batch Size

```python
# Before
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Too small!
    gradient_accumulation_steps=32
)

# After
training_args = TrainingArguments(
    per_device_train_batch_size=8,  # Better GPU utilization
    gradient_accumulation_steps=4  # Fewer accumulation steps
)
```

**Why**: Larger micro batches = better GPU utilization

---

### Tip 2: Enable Communication Overlap

```json
{
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,  // Overlap comm with computation
    "contiguous_gradients": true  // Reduce fragmentation
  }
}
```

**Expected improvement**: 10-15% faster

---

### Tip 3: Tune Bucket Sizes

```json
{
  "zero_optimization": {
    "stage": 2,
    "reduce_bucket_size": 5e8,  // Tune based on model
    "allgather_bucket_size": 5e8  // Larger = fewer comms
  }
}
```

**Guidelines**:
- Small models (< 1B): 1e8
- Medium models (1B-13B): 5e8
- Large models (> 13B): 1e9

---

### Tip 4: Use BF16 Instead of FP16

```python
training_args = TrainingArguments(
    bf16=True,  # More stable, no loss scaling needed
    deepspeed="ds_config.json"
)
```

**Benefits**:
- No loss scaling overhead
- More stable training
- Fewer NaN losses

**Requires**: Ampere GPUs (A100, H100) or newer

---

## Common Issues

### Issue 1: "DeepSpeed not installed"

**Error**:
```
ImportError: DeepSpeed is not installed. pip install deepspeed
```

**Solution**:
```bash
pip install deepspeed
```

For ZeRO-Infinity (NVMe):
```bash
DS_BUILD_AIO=1 pip install deepspeed
```

---

### Issue 2: "Batch size mismatch"

**Error**:
```
AssertionError: train_batch_size must equal micro_batch × grad_accum × num_gpus
```

**Solution**: Use "auto" in DeepSpeed config:
```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto"
}
```

---

### Issue 3: Checkpoint loading fails

**Error**:
```
RuntimeError: Cannot load checkpoint with different ZeRO stage
```

**Solution**: Use same ZeRO stage when loading:
```python
# When saving
training_args = TrainingArguments(
    output_dir="./output",
    deepspeed="ds_config_stage3.json"  # Stage 3
)
trainer.save_model()

# When loading - use SAME config
training_args = TrainingArguments(
    output_dir="./output",
    deepspeed="ds_config_stage3.json"  # Same stage!
)
trainer = Trainer(model=model, args=training_args)
```

---

### Issue 4: Training slower than expected

**Problem**: Using ZeRO-3 with small models

**Solution**: Use ZeRO-2 instead:
```json
{
  "zero_optimization": {
    "stage": 2  // Change from 3 to 2
  }
}
```

**Rule of thumb**:
- Models < 1B params: ZeRO-0 or ZeRO-1
- Models 1B-13B: ZeRO-2
- Models > 13B: ZeRO-3

---

## Best Practices

### 1. Start Simple

Begin with minimal config:
```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "fp16": {"enabled": "auto"},
  "zero_optimization": {"stage": 2}
}
```

Then add optimizations incrementally.

---

### 2. Use "auto" for Compatibility

Always use "auto" when possible to inherit from `TrainingArguments`:
```json
{
  "train_batch_size": "auto",  // ✅ Good
  "fp16": {"enabled": "auto"}  // ✅ Good
}
```

Not:
```json
{
  "train_batch_size": 128,  // ❌ Conflicts with TrainingArguments
  "fp16": {"enabled": true}  // ❌ Conflicts with TrainingArguments
}
```

---

### 3. Profile Before Optimizing

Run benchmarks to identify bottlenecks:
```bash
# Enable profiling
export DEEPSPEED_PROFILE=1

# Run training
deepspeed train.py

# Check profile
cat deepspeed_profile.json
```

---

### 4. Test on Small Model First

Before training large model:
1. Test with GPT-2 or small model
2. Verify DeepSpeed works
3. Benchmark performance
4. Then scale to large model

---

### 5. Monitor Memory Usage

```bash
# While training
watch -n 1 nvidia-smi

# Or programmatically
import torch
allocated = torch.cuda.memory_allocated() / 1e9
print(f"GPU memory: {allocated:.2f} GB")
```

---

## Complete Example

### Your Existing HF Trainer Script

```python
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2-large")
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

# Training arguments (your existing code)
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    fp16=True,
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
    deepspeed="ds_config.json",  # ← ONLY CHANGE NEEDED
)

# Trainer (your existing code)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# Train (your existing code)
trainer.train()
```

### DeepSpeed Config (ds_config.json)

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "fp16": {
    "enabled": "auto",
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000
  },
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8,
    "reduce_scatter": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }
  }
}
```

### Launch

```bash
# Single node (8 GPUs)
deepspeed --num_gpus=8 train.py

# Multi-node
deepspeed --hostfile=hostfile --master_port=29500 train.py
```

**Expected results**:
- **Memory**: 50-60% reduction vs vanilla HF Trainer
- **Speed**: 5-10% slower than vanilla (acceptable tradeoff)
- **Capability**: Can train models 2-3× larger

---

## Migration Checklist

- [ ] Install DeepSpeed: `pip install deepspeed`
- [ ] Create minimal `ds_config.json` with "auto" values
- [ ] Add `deepspeed="ds_config.json"` to `TrainingArguments`
- [ ] Test with small model (GPT-2)
- [ ] Benchmark memory usage and speed
- [ ] Choose appropriate ZeRO stage
- [ ] Add optimizations (overlap_comm, etc.)
- [ ] Test checkpointing (save and load)
- [ ] Scale to full model

---

## Additional Resources

- **[HuggingFace DeepSpeed Integration](https://huggingface.co/docs/transformers/main_classes/deepspeed)** - Official HF docs
- **[DeepSpeed Configuration](https://www.deepspeed.ai/docs/config-json/)** - Config reference
- **[ZeRO-3 Concept to Code](../guides/ZeRO3_Concept_to_Code.md)** - Deep dive
- **[Troubleshooting Guide](../guides/Troubleshooting_Guide.md)** - Common issues
- **[Performance Benchmarks](../benchmarks/README.md)** - ZeRO comparison

---

## Next Steps

1. **Run benchmarks**: Compare ZeRO stages for your model
2. **Enable offloading**: If still OOM, try CPU/NVMe offload
3. **Optimize performance**: Tune overlap_comm, bucket sizes
4. **Scale up**: Train larger models or increase batch size

**Happy training with HuggingFace + DeepSpeed!** 🚀
