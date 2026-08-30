# DeepSpeed Compression Training Tutorial

A comprehensive guide to gradient and communication compression in DeepSpeed, covering 1-bit Adam, 1-bit LAMB, and 8-bit compression for efficient multi-node training.

---

## Table of Contents

1. [Introduction to Compression](#introduction-to-compression)
2. [Why Use Compression?](#why-use-compression)
3. [1-bit Adam](#1-bit-adam)
4. [1-bit LAMB](#1-bit-lamb)
5. [8-bit Compression](#8-bit-compression)
6. [Configuration Guide](#configuration-guide)
7. [Performance Analysis](#performance-analysis)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Introduction to Compression

### What is Communication Compression?

**Communication compression** reduces the amount of data transferred between GPUs during distributed training by compressing gradients and optimizer states.

### The Communication Bottleneck

In multi-node distributed training:

```
Single Node (8 GPUs):
- NVLink: 600 GB/s between GPUs
- Communication: ~5% of training time

Multi-Node (8 nodes × 8 GPUs):
- InfiniBand: 100-200 Gb/s = 12-25 GB/s
- Communication: 40-60% of training time ← BOTTLENECK!
```

**Problem**: Inter-node bandwidth is 20-50× slower than intra-node.

**Solution**: Compress gradients before sending across nodes.

---

## Why Use Compression?

### Benefits

| Metric | Without Compression | With 1-bit Compression |
|--------|---------------------|------------------------|
| **Gradient Size** | 32 bits/param | 1 bit/param (32× smaller) |
| **Communication Time** | 100% | 10-20% |
| **Training Speed** | Baseline | 2-5× faster (multi-node) |
| **Convergence** | Baseline | Nearly identical |
| **Accuracy** | Baseline | No degradation |

### When to Use Compression

✅ **Use compression if**:
- Training across multiple nodes
- Network bandwidth < 100 Gbps
- Communication is bottleneck (>30% of time)
- Model has > 1B parameters

❌ **Skip compression if**:
- Single node training (NVLink is fast enough)
- Network bandwidth > 200 Gbps
- Small models (< 100M parameters)
- Computation is bottleneck

---

## 1-bit Adam

### How 1-bit Adam Works

**Standard Adam** (32-bit gradients):
```
1. Compute gradients: g_t
2. All-Reduce: sum gradients across GPUs (32 bits × params)
3. Update: m_t = β₁m_{t-1} + (1-β₁)g_t
           v_t = β₂v_{t-1} + (1-β₂)g_t²
           θ_t = θ_{t-1} - α·m_t / (√v_t + ε)
```

**1-bit Adam** (1-bit communication):
```
1. Compute gradients: g_t
2. Compress to 1-bit:
   - Compute E[g_t] = mean(g_t)
   - Quantize: sign(g_t) + E[g_t]
3. All-Reduce: sum compressed gradients (1 bit × params) ← 32× less data!
4. Decompress and add error compensation
5. Update with momentum and variance
```

### Key Innovation: Error Compensation

**Problem**: Quantization introduces error.

**Solution**: Track and compensate for accumulated error:

```python
# Pseudocode for 1-bit Adam
error_feedback = 0

for step in training:
    # Compute gradient
    grad = compute_gradient()

    # Add error compensation
    compensated_grad = grad + error_feedback

    # Compress to 1-bit
    mean = compensated_grad.mean()
    compressed = sign(compensated_grad) + mean

    # All-reduce (1-bit)
    all_reduced_compressed = all_reduce(compressed)

    # Decompress
    decompressed = all_reduced_compressed * abs(compensated_grad).mean()

    # Compute error for next step
    error_feedback = compensated_grad - decompressed

    # Adam update
    update_adam(decompressed)
```

---

### Enabling 1-bit Adam

#### Method 1: DeepSpeed Config (Recommended)

```json
{
  "train_batch_size": 256,
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2
  },
  "optimizer": {
    "type": "OneBitAdam",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01,
      "freeze_step": 400,
      "cuda_aware": false,
      "comm_backend_name": "nccl"
    }
  }
}
```

#### Method 2: Python API

```python
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.compression.compress import init_compression

# Create model
model = MyModel()

# Initialize DeepSpeed with 1-bit Adam
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config='ds_config_1bit.json'
)

# Training loop (no changes needed!)
for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

---

### 1-bit Adam Parameters

#### `freeze_step`
- **Description**: Warm-up steps before enabling compression
- **Default**: 400
- **Reason**: Let optimizer momentum stabilize first
- **Tuning**: Increase to 1000-2000 for large models

```json
{
  "optimizer": {
    "type": "OneBitAdam",
    "params": {
      "freeze_step": 1000
    }
  }
}
```

#### `cuda_aware`
- **Description**: Use CUDA-aware MPI for communication
- **Default**: `false`
- **When to enable**: If using CUDA-aware MPI (check with your cluster admin)

#### `comm_backend_name`
- **Description**: Communication backend (`"nccl"` or `"mpi"`)
- **Default**: `"nccl"`
- **Recommendation**: Use NCCL for NVIDIA GPUs

---

## 1-bit LAMB

### What is LAMB?

**LAMB (Layer-wise Adaptive Moments)** is an optimizer designed for large-batch training, developed by Google Brain.

**Key difference from Adam**: Adapts learning rate per layer based on weight/gradient norm ratio.

### Why 1-bit LAMB?

LAMB is ideal for:
- Very large batch sizes (64K, 128K tokens)
- Large models (BERT, GPT)
- Fast convergence with fewer steps

**1-bit LAMB** combines LAMB's large-batch benefits with compression's communication efficiency.

---

### Enabling 1-bit LAMB

```json
{
  "train_batch_size": 65536,
  "train_micro_batch_size_per_gpu": 256,
  "gradient_accumulation_steps": 32,
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2
  },
  "optimizer": {
    "type": "OneBitLamb",
    "params": {
      "lr": 1e-3,
      "weight_decay": 0.01,
      "bias_correction": true,
      "max_coeff": 0.3,
      "min_coeff": 0.01,
      "freeze_step": 1000,
      "cuda_aware": false,
      "comm_backend_name": "nccl"
    }
  }
}
```

### 1-bit LAMB Parameters

#### `max_coeff` and `min_coeff`
- **Description**: Bounds on per-layer learning rate scaling
- **Formula**: `layer_lr = global_lr × clip(||W|| / ||g||, min_coeff, max_coeff)`
- **Defaults**: `max_coeff=0.3`, `min_coeff=0.01`

#### `bias_correction`
- **Description**: Apply bias correction to momentum estimates
- **Default**: `true`
- **Recommendation**: Keep enabled

---

## 8-bit Compression

### How 8-bit Compression Works

Instead of 1-bit (sign only), use 8-bit quantization:

```
32-bit float → 8-bit integer
Compression ratio: 4×
Quality: Higher than 1-bit
```

**Quantization**:
```python
def quantize_8bit(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    scale = (max_val - min_val) / 255
    quantized = ((tensor - min_val) / scale).round().to(torch.uint8)
    return quantized, min_val, scale

def dequantize_8bit(quantized, min_val, scale):
    return quantized.float() * scale + min_val
```

---

### Enabling 8-bit Compression

```json
{
  "compression_training": {
    "weight_quantization": {
      "shared_parameters": {
        "enabled": true,
        "quantizer_kernel": true,
        "schedule_offset": 0,
        "quantize_groups": 1,
        "quantize_verbose": false,
        "quantization_type": "symmetric",
        "quantize_weight_in_forward": false,
        "rounding": "nearest",
        "fp16_mixed_quantize": {
          "enabled": false,
          "quantize_change_ratio": 0.001
        }
      },
      "different_groups": {
        "wq1": {
          "params": {
            "start_bits": 8,
            "target_bits": 8,
            "quantization_period": 0
          },
          "modules": ["all"]
        }
      }
    },
    "activation_quantization": {
      "shared_parameters": {
        "enabled": true,
        "quantization_type": "symmetric",
        "range_calibration": "dynamic",
        "schedule_offset": 0
      },
      "different_groups": {
        "aq1": {
          "params": {
            "bits": 8
          },
          "modules": ["all"]
        }
      }
    }
  }
}
```

**Note**: 8-bit compression is more complex to configure than 1-bit Adam/LAMB.

---

## Configuration Guide

### Minimal 1-bit Adam Config

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2
  },
  "optimizer": {
    "type": "OneBitAdam",
    "params": {
      "lr": 1e-4,
      "freeze_step": 1000
    }
  }
}
```

---

### Production 1-bit Adam Config

```json
{
  "train_batch_size": 512,
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 8,
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000
  },
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8
  },
  "optimizer": {
    "type": "OneBitAdam",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01,
      "freeze_step": 2000,
      "cuda_aware": false,
      "comm_backend_name": "nccl"
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 1e-4,
      "warmup_num_steps": 2000,
      "total_num_steps": 100000
    }
  }
}
```

---

### Multi-Node 1-bit LAMB Config

```json
{
  "train_batch_size": 32768,
  "train_micro_batch_size_per_gpu": 128,
  "gradient_accumulation_steps": 32,
  "gradient_clipping": 1.0,
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 1
  },
  "optimizer": {
    "type": "OneBitLamb",
    "params": {
      "lr": 5e-4,
      "weight_decay": 0.01,
      "bias_correction": true,
      "max_coeff": 0.3,
      "min_coeff": 0.01,
      "freeze_step": 1000,
      "cuda_aware": false,
      "comm_backend_name": "nccl"
    }
  }
}
```

---

## Performance Analysis

### Benchmark: BERT-Large Pre-training (16 nodes × 8 GPUs)

| Configuration | Comm. Time | Total Time | Speedup |
|---------------|------------|------------|---------|
| **Standard Adam** | 420s (58%) | 720s | 1.0× |
| **1-bit Adam** | 45s (15%) | 300s | 2.4× |
| **1-bit LAMB** | 40s (14%) | 285s | 2.5× |

**Result**: 1-bit compression provides **2.4-2.5× speedup** for multi-node training.

---

### Benchmark: GPT-3 1.3B (8 nodes × 8 GPUs)

| Configuration | Throughput | Convergence |
|---------------|------------|-------------|
| **Standard Adam** | 42K tok/s | 100% (baseline) |
| **1-bit Adam** | 95K tok/s | 98% (minor loss) |
| **8-bit Compression** | 78K tok/s | 99.5% |

**Result**:
- **1-bit Adam**: 2.3× faster with minimal accuracy loss
- **8-bit**: 1.9× faster with negligible accuracy loss

---

### Communication Reduction

| Method | Gradient Size | Reduction |
|--------|---------------|-----------|
| **FP32** | 32 bits/param | 1× (baseline) |
| **FP16** | 16 bits/param | 2× |
| **8-bit** | 8 bits/param | 4× |
| **1-bit** | 1 bit/param | 32× |

---

## Best Practices

### 1. Use Warm-up Period

**Why**: Let optimizer momentum stabilize before compressing.

```json
{
  "optimizer": {
    "type": "OneBitAdam",
    "params": {
      "freeze_step": 2000  // 2000 steps without compression
    }
  }
}
```

**Guidelines**:
- Small models (< 1B params): 400-1000 steps
- Medium models (1B-13B): 1000-2000 steps
- Large models (> 13B): 2000-5000 steps

---

### 2. Monitor Convergence

Track loss curve to ensure compression isn't degrading training:

```python
import matplotlib.pyplot as plt

# Plot with and without compression
plt.plot(steps, losses_standard, label='Standard Adam')
plt.plot(steps, losses_1bit, label='1-bit Adam')
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Convergence Comparison')
plt.savefig('convergence.png')
```

**Expected**: Curves should be nearly identical after warm-up.

---

### 3. Choose Optimizer Based on Batch Size

| Batch Size | Recommended Optimizer |
|------------|----------------------|
| Small (< 1K) | Standard Adam |
| Medium (1K-8K) | 1-bit Adam |
| Large (8K-64K) | 1-bit LAMB |
| Very Large (> 64K) | 1-bit LAMB with careful tuning |

---

### 4. Combine with ZeRO Stage 1 or 2

```json
{
  "zero_optimization": {
    "stage": 2  // Good balance with compression
  },
  "optimizer": {
    "type": "OneBitAdam"
  }
}
```

**Why**:
- ZeRO-3 may conflict with 1-bit Adam (both optimize communication)
- ZeRO-1/2 + 1-bit Adam = optimal for most use cases

---

### 5. Use BF16 for Stability

```json
{
  "bf16": {
    "enabled": true  // More stable than FP16 with compression
  },
  "optimizer": {
    "type": "OneBitAdam"
  }
}
```

---

### 6. Test on Single Node First

```bash
# Step 1: Verify on single node (8 GPUs)
deepspeed --num_gpus=8 train.py --deepspeed_config=ds_config_1bit.json

# Step 2: Scale to multi-node
deepspeed --hostfile=hostfile train.py --deepspeed_config=ds_config_1bit.json
```

**Why**: Easier to debug convergence issues on single node.

---

## Troubleshooting

### Issue 1: Loss Diverges After Enabling Compression

**Symptoms**:
```
Step 0-1000: loss = 2.5 → 2.1 (without compression)
Step 1000: Enable compression (freeze_step reached)
Step 1001-1100: loss = 2.1 → 3.8 (diverging!)
```

**Solutions**:

#### Solution A: Increase Warm-up Period
```json
{
  "optimizer": {
    "type": "OneBitAdam",
    "params": {
      "freeze_step": 5000  // Increase from 1000
    }
  }
}
```

#### Solution B: Reduce Learning Rate
```json
{
  "optimizer": {
    "type": "OneBitAdam",
    "params": {
      "lr": 5e-5  // Reduce from 1e-4
    }
  }
}
```

#### Solution C: Use 8-bit Instead of 1-bit
8-bit compression is more stable than 1-bit.

---

### Issue 2: No Speedup Observed

**Symptoms**:
```
Standard Adam: 500ms/step
1-bit Adam: 490ms/step (only 2% faster)
```

**Diagnosis**: Communication is not the bottleneck.

**Solutions**:

#### Check Communication Time
```json
{
  "wall_clock_breakdown": true  // Enable profiling
}
```

Look for `backward_allreduce_time` in output:
- If < 20% of total time → compression won't help much
- If > 40% of total time → compression should help significantly

#### Use More Nodes
Compression benefits increase with more nodes:
- 2 nodes: ~1.2× speedup
- 4 nodes: ~1.5× speedup
- 8+ nodes: ~2-3× speedup

---

### Issue 3: "OneBitAdam not available"

**Error**:
```
ImportError: OneBitAdam is not available
```

**Solution**: Install DeepSpeed with 1-bit Adam support:

```bash
# Option 1: Install from PyPI
pip install deepspeed

# Option 2: Build from source with 1-bit support
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
DS_BUILD_OPS=1 pip install .
```

Verify:
```python
import deepspeed
print(deepspeed.ops.op_builder.CPUAdamBuilder().is_compatible())
```

---

### Issue 4: Slow Convergence with 1-bit LAMB

**Symptoms**:
```
Standard LAMB: Reaches target loss at step 10K
1-bit LAMB: Reaches target loss at step 15K (50% more steps)
```

**Solutions**:

#### Solution A: Tune Coefficients
```json
{
  "optimizer": {
    "type": "OneBitLamb",
    "params": {
      "max_coeff": 0.5,  // Increase from 0.3
      "min_coeff": 0.005  // Reduce from 0.01
    }
  }
}
```

#### Solution B: Increase Batch Size
LAMB works best with very large batches:

```json
{
  "train_batch_size": 65536,  // Increase
  "optimizer": {
    "type": "OneBitLamb",
    "params": {
      "lr": 1e-3  // Can use higher LR with larger batches
    }
  }
}
```

---

## Advanced Topics

### Hierarchical Compression

For extremely large clusters, use hierarchical all-reduce:

```
Node 0: GPUs 0-7
Node 1: GPUs 8-15
...

Step 1: Intra-node all-reduce (NVLink, uncompressed)
Step 2: Inter-node all-reduce (InfiniBand, compressed)
Step 3: Broadcast back within node
```

DeepSpeed automatically uses hierarchical communication when beneficial.

---

### Custom Compression

Implement custom compression:

```python
from deepspeed.compression import Compressor

class MyCompressor(Compressor):
    def compress(self, tensor):
        # Your compression logic
        compressed = my_quantization(tensor)
        return compressed

    def decompress(self, compressed):
        # Your decompression logic
        tensor = my_dequantization(compressed)
        return tensor

# Register compressor
deepspeed.compression.register_compressor('my_compressor', MyCompressor)
```

---

## Complete Example: BERT Pre-training with 1-bit Adam

```python
import torch
import deepspeed
from transformers import BertForPreTraining, BertConfig

def train():
    # Model
    config = BertConfig(vocab_size=30522, hidden_size=1024, num_hidden_layers=24)
    model = BertForPreTraining(config)

    # DeepSpeed config with 1-bit Adam
    ds_config = {
        "train_batch_size": 512,
        "train_micro_batch_size_per_gpu": 8,
        "gradient_accumulation_steps": 8,
        "gradient_clipping": 1.0,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 16
        },
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True
        },
        "optimizer": {
            "type": "OneBitAdam",
            "params": {
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
                "freeze_step": 2000
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1e-4,
                "warmup_num_steps": 2000
            }
        }
    }

    # Initialize DeepSpeed
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    # Training loop
    model_engine.train()
    for step, batch in enumerate(dataloader):
        # Move to device
        batch = {k: v.to(model_engine.device) for k, v in batch.items()}

        # Forward
        outputs = model_engine(**batch)
        loss = outputs.loss

        # Backward and step
        model_engine.backward(loss)
        model_engine.step()

        # Log
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

        # Note: Compression enabled automatically at step 2000 (freeze_step)

    # Save checkpoint
    model_engine.save_checkpoint('checkpoints', tag='final')

if __name__ == '__main__':
    train()
```

---

## Summary

### When to Use Each Compression Method

| Method | Best For | Compression | Quality |
|--------|----------|-------------|---------|
| **No Compression** | Single node | 1× | 100% |
| **8-bit** | 2-4 nodes, quality-critical | 4× | 99.5% |
| **1-bit Adam** | 4-16 nodes, balanced | 32× | 98-99% |
| **1-bit LAMB** | 16+ nodes, large batch | 32× | 97-99% |

### Key Takeaways

1. **Compression for multi-node**: Essential for 8+ nodes
2. **Warm-up period**: Critical for stability
3. **Monitor convergence**: Ensure no quality degradation
4. **Combine with ZeRO-2**: Best balance
5. **Test incrementally**: Single node → multi-node

---

## Additional Resources

- **[1-bit Adam Paper](https://arxiv.org/abs/2102.02888)** - Original research
- **[1-bit LAMB Paper](https://arxiv.org/abs/2104.06069)** - LAMB compression
- **[DeepSpeed Compression](https://www.deepspeed.ai/tutorials/compressed-training/)** - Official tutorial
- **[ZeRO + Compression](https://www.deepspeed.ai/tutorials/zero-one-adam/)** - Combining techniques

**Happy compressed training!** 🚀
