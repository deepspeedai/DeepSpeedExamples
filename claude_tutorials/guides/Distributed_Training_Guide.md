# Distributed Training Data Flow Guide

## Complete Data Flow for ZeRO-3 Multi-GPU Training

This guide provides a detailed walkthrough of a single gradient descent step in a ZeRO-3 enabled multi-GPU training run, illustrating the flow of data (parameters, activations, gradients) as it is sharded, reduced, and applied across the distributed worker group.

---

## Table of Contents
1. [Setup and Initialization](#setup-and-initialization)
2. [Single Training Step Overview](#single-training-step-overview)
3. [Detailed Forward Pass](#detailed-forward-pass)
4. [Detailed Backward Pass](#detailed-backward-pass)
5. [Optimizer Step](#optimizer-step)
6. [Communication Patterns](#communication-patterns)
7. [Memory States](#memory-states)
8. [Comparison: ZeRO-1, ZeRO-2, ZeRO-3](#comparison-zero-1-zero-2-zero-3)
9. [Debugging and Monitoring](#debugging-and-monitoring)

---

## Setup and Initialization

### System Configuration
- **Model**: 2-layer Transformer (simplified)
  - Layer 1: 1000 parameters
  - Layer 2: 1000 parameters
  - **Total**: 2000 parameters
- **GPUs**: 4 (GPU 0, 1, 2, 3)
- **Batch size per GPU**: 2 samples
- **Data type**: BF16 (2 bytes per parameter)
- **Optimizer**: Adam (requires momentum + variance states)

### Initial State After deepspeed.initialize()

```
ZeRO-3 Initialization:
├─ Parameters partitioned across 4 GPUs
│  GPU 0 owns: params[0:500]      (500 params, 1KB)
│  GPU 1 owns: params[500:1000]   (500 params, 1KB)
│  GPU 2 owns: params[1000:1500]  (500 params, 1KB)
│  GPU 3 owns: params[1500:2000]  (500 params, 1KB)
│
├─ Optimizer states partitioned
│  GPU 0: momentum[0:500], variance[0:500]
│  GPU 1: momentum[500:1000], variance[500:1000]
│  GPU 2: momentum[1000:1500], variance[1000:1500]
│  GPU 3: momentum[1500:2000], variance[1500:2000]
│
└─ Each GPU loads its own data batch
   GPU 0: batch[0:2]
   GPU 1: batch[2:4]
   GPU 2: batch[4:6]
   GPU 3: batch[6:8]
```

**Memory Footprint**:
- Parameters per GPU: 2000/4 = 500 params × 2 bytes = 1KB
- Optimizer states per GPU: 500 × 8 bytes (fp32) = 4KB
- Total per GPU: ~5KB (vs 2000 × 2 = 4KB parameters in standard data parallel)

---

## Single Training Step Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     ONE TRAINING STEP                            │
└─────────────────────────────────────────────────────────────────┘

1. FORWARD PASS
   ├─ For each layer:
   │  ├─ All-Gather parameters (GPU → GPU communication)
   │  ├─ Forward computation
   │  └─ Release parameters
   └─ Compute loss

2. BACKWARD PASS
   ├─ For each layer (reverse order):
   │  ├─ All-Gather parameters (GPU → GPU communication)
   │  ├─ Backward computation (compute gradients)
   │  ├─ Reduce-Scatter gradients (GPU → GPU communication)
   │  └─ Release parameters
   └─ All gradients computed and partitioned

3. OPTIMIZER STEP
   ├─ Each GPU updates its parameter partition independently
   ├─ No communication needed
   └─ Model ready for next forward pass
```

---

## Detailed Forward Pass

### Layer 1 Forward Pass

#### Step 1: Pre-Forward Hook - All-Gather Parameters

```
BEFORE All-Gather (each GPU has its partition):
GPU 0: [W1[0:250]]
GPU 1: [W1[250:500]]
GPU 2: [W1[500:750], b1[0:100]]
GPU 3: [W1[750:1000], b1[100:200]]

All-Gather Operation:
GPU 0 sends W1[0:250]     → to all GPUs
GPU 1 sends W1[250:500]   → to all GPUs
GPU 2 sends W1[500:750]   → to all GPUs
GPU 3 sends W1[750:1000]  → to all GPUs

AFTER All-Gather (each GPU has full Layer 1):
GPU 0: [W1[0:1000], b1[0:200]] ← Full layer!
GPU 1: [W1[0:1000], b1[0:200]]
GPU 2: [W1[0:1000], b1[0:200]]
GPU 3: [W1[0:1000], b1[0:200]]
```

**Communication Details**:
- **Collective**: NCCL All-Gather
- **Data transferred per GPU**: 1000 params × 2 bytes × 3/4 = 1.5KB (send my 1/4, receive other 3/4)
- **Time**: ~10-100 microseconds (depends on GPU interconnect)

#### Step 2: Forward Computation

Each GPU independently computes forward pass with its own batch:

```
GPU 0:
  input: batch[0:2]  (2 samples)
  compute: output = Layer1(input, W1, b1)
  result: activations[0:2]

GPU 1:
  input: batch[2:4]
  compute: output = Layer1(input, W1, b1)
  result: activations[2:4]

GPU 2:
  input: batch[4:6]
  compute: output = Layer1(input, W1, b1)
  result: activations[4:6]

GPU 3:
  input: batch[6:8]
  compute: output = Layer1(input, W1, b1)
  result: activations[6:8]
```

**Note**: Same parameters (W1, b1), different inputs → different activations.

#### Step 3: Post-Forward Hook - Release Parameters

```
Each GPU releases gathered parameters, keeping only its partition:

GPU 0: [W1[0:250]]           ← Back to partition
GPU 1: [W1[250:500]]
GPU 2: [W1[500:750], b1[0:100]]
GPU 3: [W1[750:1000], b1[100:200]]
```

**Memory Freed**: 3/4 of layer parameters (750 params × 2 bytes = 1.5KB per GPU)

### Layer 2 Forward Pass

**Repeat the same process for Layer 2**:
1. All-Gather Layer 2 parameters
2. Compute Layer 2 forward with Layer 1 activations as input
3. Release Layer 2 parameters

### Loss Computation

```
Each GPU computes loss for its batch:

GPU 0: loss_0 = criterion(output[0:2], target[0:2])
GPU 1: loss_1 = criterion(output[2:4], target[2:4])
GPU 2: loss_2 = criterion(output[4:6], target[4:6])
GPU 3: loss_3 = criterion(output[6:8], target[6:8])
```

---

## Detailed Backward Pass

### Loss Backward

```
Each GPU computes gradient of loss wrt output:

GPU 0: grad_output[0:2] = ∂loss_0/∂output[0:2]
GPU 1: grad_output[2:4] = ∂loss_1/∂output[2:4]
GPU 2: grad_output[4:6] = ∂loss_2/∂output[4:6]
GPU 3: grad_output[6:8] = ∂loss_3/∂output[6:8]
```

### Layer 2 Backward Pass

#### Step 1: Pre-Backward Hook - All-Gather Parameters (Again!)

```
BEFORE All-Gather:
GPU 0: [W2[0:250]]
GPU 1: [W2[250:500]]
GPU 2: [W2[500:750]]
GPU 3: [W2[750:1000]]

AFTER All-Gather:
GPU 0: [W2[0:1000], b2[0:200]] ← Full layer
GPU 1: [W2[0:1000], b2[0:200]]
GPU 2: [W2[0:1000], b2[0:200]]
GPU 3: [W2[0:1000], b2[0:200]]
```

**Why gather again?** Need full parameters to compute gradients correctly.

#### Step 2: Backward Computation

Each GPU computes gradients for its batch:

```
GPU 0:
  grad_input[0:2] = ∂loss_0/∂input[0:2]
  grad_W2_full = ∂loss_0/∂W2     ← Full gradient for W2
  grad_b2_full = ∂loss_0/∂b2     ← Full gradient for b2

GPU 1:
  grad_input[2:4] = ∂loss_1/∂input[2:4]
  grad_W2_full = ∂loss_1/∂W2     ← Full gradient for W2
  grad_b2_full = ∂loss_1/∂b2

GPU 2:
  grad_input[4:6] = ∂loss_2/∂input[4:6]
  grad_W2_full = ∂loss_2/∂W2
  grad_b2_full = ∂loss_2/∂b2

GPU 3:
  grad_input[6:8] = ∂loss_3/∂input[6:8]
  grad_W2_full = ∂loss_3/∂W2
  grad_b2_full = ∂loss_3/∂b2
```

**Key Point**: Each GPU computes **full** gradients for parameters (based on its batch).

#### Step 3: Post-Backward Hook - Reduce-Scatter Gradients

**Reduce-Scatter** combines two operations:
1. **Reduce** (sum): Sum gradients across all GPUs
2. **Scatter** (partition): Distribute summed gradients so each GPU gets its partition

```
BEFORE Reduce-Scatter (each GPU has full gradients):
GPU 0: [grad_W2_0[0:1000], grad_b2_0[0:200]]
GPU 1: [grad_W2_1[0:1000], grad_b2_1[0:200]]
GPU 2: [grad_W2_2[0:1000], grad_b2_2[0:200]]
GPU 3: [grad_W2_3[0:1000], grad_b2_3[0:200]]

Reduce-Scatter Operation:
Step 1 (Reduce): Sum gradients element-wise
  grad_W2_sum = grad_W2_0 + grad_W2_1 + grad_W2_2 + grad_W2_3
  grad_b2_sum = grad_b2_0 + grad_b2_1 + grad_b2_2 + grad_b2_3

Step 2 (Scatter): Partition and distribute
  Send grad_W2_sum[0:250]   → GPU 0
  Send grad_W2_sum[250:500] → GPU 1
  Send grad_W2_sum[500:750] → GPU 2
  Send grad_W2_sum[750:1000] + grad_b2_sum → GPU 3

AFTER Reduce-Scatter (each GPU has its gradient partition):
GPU 0: [grad_sum[0:250]]
GPU 1: [grad_sum[250:500]]
GPU 2: [grad_sum[500:750]]
GPU 3: [grad_sum[750:1000], grad_b2_sum]
```

**Communication Details**:
- **Collective**: NCCL Reduce-Scatter
- **Data transferred per GPU**: Same as All-Gather (~1.5KB)
- **Result**: Each GPU has gradients only for its parameter partition

#### Step 4: Release Parameters

```
GPU 0: [W2[0:250]]           ← Back to partition
GPU 1: [W2[250:500]]
GPU 2: [W2[500:750]]
GPU 3: [W2[750:1000]]
```

### Layer 1 Backward Pass

**Repeat the same process** for Layer 1 (in reverse topological order).

---

## Optimizer Step

### State After Backward Pass

```
Each GPU has:
├─ Parameter partition (owned)
├─ Gradient partition (summed across all GPUs)
└─ Optimizer state partition (momentum, variance)

GPU 0:
  params[0:250]
  grads[0:250]        ← Sum of gradients from all GPUs
  momentum[0:250]
  variance[0:250]

GPU 1:
  params[250:500]
  grads[250:500]
  momentum[250:500]
  variance[250:500]

GPU 2:
  params[500:750]
  grads[500:750]
  momentum[500:750]
  variance[500:750]

GPU 3:
  params[750:1000]
  grads[750:1000]
  momentum[750:1000]
  variance[750:1000]
```

### Adam Optimizer Update (Simplified)

Each GPU independently updates its parameter partition:

```python
# GPU 0 updates params[0:250]
for i in range(0, 250):
    momentum[i] = beta1 * momentum[i] + (1 - beta1) * grads[i]
    variance[i] = beta2 * variance[i] + (1 - beta2) * grads[i]**2
    params[i] = params[i] - lr * momentum[i] / (sqrt(variance[i]) + eps)

# GPU 1 updates params[250:500]
for i in range(250, 500):
    momentum[i] = beta1 * momentum[i] + (1 - beta1) * grads[i]
    variance[i] = beta2 * variance[i] + (1 - beta2) * grads[i]**2
    params[i] = params[i] - lr * momentum[i] / (sqrt(variance[i]) + eps)

# GPU 2 updates params[500:750]
# GPU 3 updates params[750:1000]
# ... same pattern
```

**Key Properties**:
1. **No communication** - Each GPU works on its partition independently
2. **Full model updated** - Collectively, all GPUs update all parameters
3. **Ready for next forward** - Next forward will All-Gather updated parameters

---

## Communication Patterns

### Summary of Collective Operations

| Phase | Operation | Direction | Data Size | Purpose |
|-------|-----------|-----------|-----------|---------|
| Forward (per layer) | All-Gather | All → All | (N-1)/N × layer_params | Gather full parameters |
| Backward (per layer) | All-Gather | All → All | (N-1)/N × layer_params | Gather parameters for gradients |
| Backward (per layer) | Reduce-Scatter | All → All | (N-1)/N × layer_params | Sum and partition gradients |
| Optimizer | None | - | 0 | Local updates only |

**Total Communication per Step**:
- All-Gather (forward): P × 2 bytes × (N-1)/N × num_layers
- All-Gather (backward): P × 2 bytes × (N-1)/N × num_layers
- Reduce-Scatter: P × 2 bytes × (N-1)/N × num_layers
- **Total**: ~3 × P × 2 bytes (where P = total parameters)

### Communication Topology

```
Ring All-Gather (most common for NCCL):

Step 1: GPU i sends to GPU (i+1) % N
  GPU 0 → GPU 1
  GPU 1 → GPU 2
  GPU 2 → GPU 3
  GPU 3 → GPU 0

Step 2: GPU i sends to GPU (i+1) % N (different chunks)
  ... (N-1 steps total)

Result: All GPUs have full data with optimal bandwidth usage
```

### Overlap Optimization

**Without Overlap**:
```
[Gather Layer 1] → [Compute Layer 1] → [Gather Layer 2] → [Compute Layer 2]
     100μs            500μs              100μs              500μs
                                        ↑ GPU idle
```

**With Overlap** (`overlap_comm: true`):
```
[Gather Layer 1] → [Compute Layer 1]   [Compute Layer 2]
     100μs         ↓ [Gather Layer 2] ↓    500μs
                     (overlapped!)
```

---

## Memory States

### Memory Timeline for One GPU (GPU 0)

```
Time     |  Parameters  |  Gradients  |  Activations  |  Total
---------|--------------|-------------|---------------|--------
T0       |  0.5KB (1/4) |  0          |  0            |  0.5KB
(init)   |              |             |               |
---------|--------------|-------------|---------------|--------
T1       |  2KB (full)  |  0          |  5KB          |  7KB
(L1 fwd) |  ← gathered  |             |  ← computed   |
---------|--------------|-------------|---------------|--------
T2       |  0.5KB (1/4) |  0          |  5KB          |  5.5KB
(L1 end) |  ← released  |             |               |
---------|--------------|-------------|---------------|--------
T3       |  2KB (full)  |  0          |  10KB         |  12KB
(L2 fwd) |  ← gathered  |             |  ← L1+L2 act  |
---------|--------------|-------------|---------------|--------
T4       |  0.5KB (1/4) |  0          |  10KB         |  10.5KB
(L2 end) |  ← released  |             |               |
---------|--------------|-------------|---------------|--------
T5       |  2KB (full)  |  0.5KB (1/4)|  5KB          |  7.5KB
(L2 bwd) |  ← gathered  |  ← computed |  ← L1 only    |
---------|--------------|-------------|---------------|--------
T6       |  0.5KB (1/4) |  0.5KB (1/4)|  5KB          |  6KB
(L2 end) |  ← released  |             |               |
---------|--------------|-------------|---------------|--------
T7       |  2KB (full)  |  1KB (1/2)  |  0            |  3KB
(L1 bwd) |  ← gathered  |  ← L1+L2    |  ← released   |
---------|--------------|-------------|---------------|--------
T8       |  0.5KB (1/4) |  0.5KB (1/4)|  0            |  1KB
(L1 end) |  ← released  |  ← RS       |               |
---------|--------------|-------------|---------------|--------
T9       |  0.5KB (1/4) |  0          |  0            |  0.5KB
(optim)  |              |  ← zeroed   |               |

Peak Memory: T3 = 12KB (during L2 forward with full params + all activations)
```

**Key Observations**:
1. Peak memory is determined by the largest layer + activations
2. Parameters are gathered and released continuously
3. Activations can be further reduced with gradient checkpointing

### With Activation Checkpointing

Activation checkpointing discards activations during forward, recomputes during backward:

```
Without checkpointing: Store all activations → More memory
With checkpointing: Recompute activations → Less memory, more compute

Memory savings: ~30-40% reduction in activation memory
Compute overhead: ~30% slower training
```

---

## Comparison: ZeRO-1, ZeRO-2, ZeRO-3

### What Gets Partitioned

| Stage | Parameters | Gradients | Optimizer States | Memory Savings |
|-------|------------|-----------|------------------|----------------|
| ZeRO-1 | ✗ Full | ✗ Full | ✓ Partitioned | ~4× |
| ZeRO-2 | ✗ Full | ✓ Partitioned | ✓ Partitioned | ~8× |
| ZeRO-3 | ✓ Partitioned | ✓ Partitioned | ✓ Partitioned | ~N× |

### Communication Patterns

#### ZeRO-1 (Optimizer State Partitioning)

```
Forward:  No communication (full model on each GPU)
Backward: All-Reduce gradients (sum across GPUs)
Optimizer: All-Gather updated parameters
```

#### ZeRO-2 (Optimizer + Gradient Partitioning)

```
Forward:  No communication (full model on each GPU)
Backward: Reduce-Scatter gradients (sum and partition)
Optimizer: All-Gather updated parameters
```

#### ZeRO-3 (Full Partitioning)

```
Forward:  All-Gather parameters (per layer)
Backward: All-Gather parameters + Reduce-Scatter gradients (per layer)
Optimizer: No communication
```

### Memory Breakdown Example

**Model**: 7B parameters, 8 GPUs, BF16 training, Adam optimizer

| Component | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|-----------|--------|--------|--------|
| Parameters | 14GB | 14GB | 1.75GB |
| Gradients | 14GB | 1.75GB | 1.75GB |
| Optimizer States | 10.5GB | 10.5GB | 10.5GB |
| **Total per GPU** | **38.5GB** | **26.25GB** | **14GB** |

*Note: ZeRO-3 "14GB" is worst-case with all layers' parameters in memory; actual peak is much lower due to gather/release.*

---

## Debugging and Monitoring

### Logging Parameter States

```python
import deepspeed

# After model initialization
for name, param in model.named_parameters():
    if hasattr(param, 'ds_status'):
        print(f"{name}:")
        print(f"  Status: {param.ds_status}")
        print(f"  Shape: {param.ds_shape}")
        print(f"  Partition shape: {param.ds_tensor.shape}")
        print(f"  Owner rank: {param.ds_process_group}")
```

**Possible states**:
- `NOT_AVAILABLE`: Parameter is partitioned, not in GPU memory
- `AVAILABLE`: Parameter is gathered and available for computation
- `INFLIGHT`: Parameter is being gathered/released

### Monitoring Communication

```bash
# Monitor GPU utilization and communication
nvidia-smi dmon -i 0 -s pucvmet -d 1

# Monitor network utilization (for multi-node)
iftop -i ib0  # InfiniBand interface
```

**Metrics to watch**:
- **GPU Utilization**: Should be >80% if communication is well-overlapped
- **GPU Memory**: Should stay below limit with ZeRO-3
- **Network Bandwidth**: Should be saturated during All-Gather

### Common Issues

#### Issue 1: Out of Memory (OOM)

**Symptoms**: CUDA OOM error during training

**Diagnosis**:
```python
# Check peak memory
import torch
print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Check configuration
print(f"Max live params: {ds_config['zero_optimization']['stage3_max_live_parameters']}")
```

**Solutions**:
- Reduce `stage3_max_live_parameters`
- Enable `offload_param` to CPU
- Enable activation checkpointing
- Reduce batch size

#### Issue 2: Slow Training

**Symptoms**: Low GPU utilization (<50%)

**Diagnosis**:
```bash
# Check if communication is bottleneck
nvidia-smi dmon -i 0 -s u
# If utilization low, communication is likely bottleneck
```

**Solutions**:
- Enable `overlap_comm: true`
- Increase `stage3_prefetch_bucket_size`
- Use gradient accumulation
- Check network bandwidth (multi-node)

#### Issue 3: Incorrect Results

**Symptoms**: Loss is NaN or diverges

**Diagnosis**:
- Check for gradient clipping
- Verify data loading (all GPUs should have different batches)
- Check learning rate

**Solutions**:
- Enable `gradient_clipping: 1.0`
- Verify DistributedSampler is used
- Reduce learning rate

---

## Summary

### Single Step Data Flow

1. **Forward Pass**:
   - For each layer: All-Gather params → Compute → Release params
   - Result: Activations stored, parameters partitioned

2. **Backward Pass**:
   - For each layer (reverse): All-Gather params → Compute gradients → Reduce-Scatter grads → Release params
   - Result: Gradients partitioned and summed across GPUs

3. **Optimizer Step**:
   - Each GPU updates its parameter partition independently
   - No communication needed

### Key Insights

1. **Trade Memory for Communication**: ZeRO-3 uses ~N× less memory but requires ~3× more communication
2. **On-Demand Gathering**: Parameters are gathered only when needed, immediately released
3. **Independent Optimizer**: Each GPU maintains its partition independently
4. **Overlap is Critical**: Communication must overlap computation for good performance

### Communication Volume

For P parameters on N GPUs:
- **ZeRO-1**: P × 2 bytes (All-Reduce gradients)
- **ZeRO-2**: P × 2 bytes (Reduce-Scatter + All-Gather)
- **ZeRO-3**: ~3 × P × 2 bytes (multiple All-Gather + Reduce-Scatter per layer)

### When to Use ZeRO-3

✅ Model > GPU memory
✅ Multiple GPUs with fast interconnect
✅ Can tolerate some slowdown for larger models

❌ Model fits in single GPU
❌ Few GPUs with slow interconnect
❌ Latency-critical applications

---

*For implementation details, see `ZeRO3_Concept_to_Code.md`*
*For practical examples, see annotated scripts in `../annotated_scripts/`*
