# ZeRO-3 Concept-to-Code Reference Guide

## Table of Contents
1. [Overview](#overview)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Parameter Partitioning](#parameter-partitioning)
4. [All-Gather Operations](#all-gather-operations)
5. [Forward Pass](#forward-pass)
6. [Backward Pass](#backward-pass)
7. [Optimizer Step](#optimizer-step)
8. [Source Code Mapping](#source-code-mapping)
9. [Configuration Parameters](#configuration-parameters)
10. [Performance Considerations](#performance-considerations)

---

## Overview

**ZeRO-3** (Zero Redundancy Optimizer, Stage 3) is DeepSpeed's most aggressive memory optimization technique, enabling training of models that are significantly larger than GPU memory.

### Key Concept
Instead of replicating the entire model on each GPU (standard data parallelism), ZeRO-3 partitions ALL model states across GPUs:
- **Parameters (weights)**
- **Gradients**
- **Optimizer states (momentum, variance, etc.)**

### Memory Savings
For a model with P parameters trained on N GPUs:
- **Standard Data Parallel**: Each GPU stores P parameters + P gradients + 2P optimizer states (for Adam) = 4P per GPU
- **ZeRO-3**: Each GPU stores P/N parameters + P/N gradients + 2P/N optimizer states = 4P/N per GPU

**Result**: N× memory reduction, enabling N× larger models!

---

## Theoretical Foundation

### The ZeRO Paper
**Paper**: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (Rajbhandari et al., 2020)
**arXiv**: https://arxiv.org/abs/1910.02054

### Three Stages of ZeRO

| Stage | What's Partitioned | Memory Saved | Communication Added |
|-------|-------------------|--------------|---------------------|
| ZeRO-1 | Optimizer states only | ~4× | Minimal |
| ZeRO-2 | Optimizer + Gradients | ~8× | Moderate (Reduce-Scatter) |
| ZeRO-3 | Optimizer + Gradients + Parameters | ~N× | Significant (All-Gather) |

### ZeRO-3 Core Idea

**Problem**: Model parameters are replicated on each GPU in standard data parallelism.

**Solution**:
1. Partition parameters across GPUs (each GPU owns 1/N of the model)
2. **Gather parameters on-demand** when needed for computation
3. **Release parameters immediately** after use
4. Result: Only one layer's parameters in memory at a time!

**Analogy**: Like a library where books (parameters) are distributed across N shelves (GPUs). When you need a book, you temporarily gather all relevant pages from all shelves, use them, then return them.

---

## Parameter Partitioning

### Conceptual Model

Consider a simple 2-layer network with 4 GPUs:

```
Original Model (on each GPU in standard data parallel):
Layer 1: [W1: 1000 params] [b1: 100 params]
Layer 2: [W2: 2000 params] [b2: 200 params]
Total: 3300 params per GPU

ZeRO-3 Partitioned (each GPU stores 1/4):
GPU 0: [W1[0:250]]
GPU 1: [W1[250:500]]
GPU 2: [W1[500:750], b1[0:100], W2[0:150]]
GPU 3: [W2[150:400]]
... and so on

Total: 825 params per GPU (4× reduction!)
```

### How Partitioning Works

1. **Flatten all parameters** into a 1D array
2. **Divide into N equal chunks** (N = number of GPUs)
3. **Assign each chunk to a GPU**
4. **Each GPU becomes the owner** of its chunk

### Owner Responsibilities
- Store the parameter partition on GPU
- Update the partition during optimizer step
- Provide the partition when others request it (All-Gather)

---

## All-Gather Operations

### What is All-Gather?

**All-Gather** is a collective communication primitive that gathers data from all processes and distributes the complete result to all processes.

**Example with 4 GPUs**:
```
Input (each GPU has different data):
GPU 0: [A0]
GPU 1: [A1]
GPU 2: [A2]
GPU 3: [A3]

After All-Gather (each GPU has full data):
GPU 0: [A0, A1, A2, A3]
GPU 1: [A0, A1, A2, A3]
GPU 2: [A0, A1, A2, A3]
GPU 3: [A0, A1, A2, A3]
```

### All-Gather in ZeRO-3

**Purpose**: Reconstruct full parameters from partitions before computation

**Workflow**:
```python
# Each GPU has 1/N of parameters
# Before computing layer:
full_params = all_gather(my_partition)  # Gather from all GPUs
output = layer_forward(full_params, input)  # Compute with full params
del full_params  # Release immediately to save memory
```

**Communication Cost**:
- Data transferred: (N-1)/N × P bytes (where P = parameter size)
- Time complexity: O(P) with ring all-gather algorithm
- Example: 1B parameters (2GB BF16) on 8 GPUs = 1.75GB transferred per GPU

---

## Forward Pass

### Standard Forward Pass (without ZeRO-3)
```python
# All layers' parameters already on GPU
x = input
for layer in model.layers:
    x = layer(x)  # Full parameters available
output = x
```

### ZeRO-3 Forward Pass
```python
x = input
for layer in model.layers:
    # 1. Pre-forward hook: Gather parameters for this layer
    full_params = all_gather(layer.partition)

    # 2. Forward computation
    x = layer.forward(x, full_params)

    # 3. Post-forward hook: Release parameters
    del full_params  # Keep only the 1/N partition

output = x
```

### Key Insight
**Only one layer's full parameters in GPU memory at any time!**

This is how ZeRO-3 trains models larger than GPU memory - it trades memory for communication.

### Memory Timeline
```
Time →
T0: [Layer 1 full params] [Layer 1 forward]
T1: [Release Layer 1, gather Layer 2 full params] [Layer 2 forward]
T2: [Release Layer 2, gather Layer 3 full params] [Layer 3 forward]
...

Peak memory: Max(layer_params) instead of Sum(all_layer_params)
```

---

## Backward Pass

### Standard Backward Pass
```python
loss.backward()
# All gradients computed and stored
# Then: gradient sync via all-reduce
```

### ZeRO-3 Backward Pass

The backward pass has the same pattern as forward, but in reverse:

```python
# Starting from loss
loss.backward()

# For each layer in reverse order:
for layer in reversed(model.layers):
    # 1. Pre-backward hook: Gather parameters again
    #    (needed for gradient computation)
    full_params = all_gather(layer.partition)

    # 2. Compute gradients
    #    Gradient wrt inputs and parameters
    gradients = backward_pass(layer, full_params)

    # 3. Post-backward hook:
    #    a) Reduce-Scatter: Sum gradients across GPUs and partition
    my_grad_partition = reduce_scatter(gradients)

    #    b) Release parameters
    del full_params

    #    c) Store my gradient partition only
    layer.grad = my_grad_partition
```

### Reduce-Scatter Operation

**Reduce-Scatter** = Reduce (sum) + Scatter (partition)

**Example with 4 GPUs**:
```
Input (each GPU computed full gradients):
GPU 0: [G0, G1, G2, G3]  (where Gi are gradient chunks)
GPU 1: [G0, G1, G2, G3]
GPU 2: [G0, G1, G2, G3]
GPU 3: [G0, G1, G2, G3]

After Reduce-Scatter:
GPU 0: [Sum(G0)]  ← owns G0 partition
GPU 1: [Sum(G1)]  ← owns G1 partition
GPU 2: [Sum(G2)]  ← owns G2 partition
GPU 3: [Sum(G3)]  ← owns G3 partition
```

### Why Reduce-Scatter?
1. **Sum gradients** across GPUs (data parallel gradient averaging)
2. **Partition summed gradients** so each GPU gets gradients for its parameter partition only
3. Saves memory: Each GPU stores 1/N of gradients instead of full gradients

---

## Optimizer Step

### Standard Optimizer Step
```python
# Each GPU has full model and full gradients
for param, grad in zip(model.parameters(), gradients):
    # Adam update (simplified)
    param.momentum = beta1 * param.momentum + (1-beta1) * grad
    param.variance = beta2 * param.variance + (1-beta2) * grad**2
    param.data -= lr * param.momentum / (sqrt(param.variance) + eps)
```

### ZeRO-3 Optimizer Step

```python
# Each GPU has 1/N of parameters and 1/N of gradients
for param_partition, grad_partition in zip(my_params, my_grads):
    # Update only my partition
    param.momentum = beta1 * param.momentum + (1-beta1) * grad_partition
    param.variance = beta2 * param.variance + (1-beta2) * grad_partition**2
    param_partition -= lr * param.momentum / (sqrt(param.variance) + eps)

# No communication needed!
# Each GPU independently updates its parameter partition
```

### Key Properties
1. **No communication during optimizer step** (each GPU updates its partition independently)
2. **No parameter all-gather needed** (next forward pass will gather updated partitions)
3. **Optimizer states are also partitioned** (1/N memory usage)

### With CPU Offload

```python
# Optimizer states are on CPU
for param_partition, grad_partition in zip(my_params, my_grads):
    # 1. Transfer gradients to CPU
    grad_cpu = grad_partition.to('cpu')

    # 2. Update on CPU (optimizer states already on CPU)
    param_cpu = update_on_cpu(param_cpu, grad_cpu, momentum_cpu, variance_cpu)

    # 3. Transfer updated parameters back to GPU
    param_partition.copy_(param_cpu.to('cuda'))
```

---

## Source Code Mapping

### Critical DeepSpeed Files

Here are the 3 most important files implementing ZeRO-3:

#### 1. **`deepspeed/runtime/zero/stage3.py`**
   - **Path**: `deepspeed/runtime/zero/stage3.py`
   - **Purpose**: Main ZeRO-3 orchestration logic
   - **Key Classes**:
     - `DeepSpeedZeroOptimizer_Stage3`: Main ZeRO-3 optimizer wrapper
   - **Key Methods**:
     - `step()`: Optimizer step with partitioned parameters
     - `backward()`: Backward pass coordination
     - `_partition_gradients()`: Partition gradients after backward

#### 2. **`deepspeed/runtime/zero/partition_parameters.py`**
   - **Path**: `deepspeed/runtime/zero/partition_parameters.py`
   - **Purpose**: Parameter partitioning and gathering logic
   - **Key Classes**:
     - `Init`: Context manager for initializing partitioned parameters
   - **Key Functions**:
     - `_partition_param()`: Partition a parameter across GPUs
     - `_all_gather_params()`: Gather full parameters from all GPUs

#### 3. **`deepspeed/runtime/zero/partitioned_param_coordinator.py`**
   - **Path**: `deepspeed/runtime/zero/partitioned_param_coordinator.py`
   - **Purpose**: Coordinates All-Gather operations during forward/backward
   - **Key Classes**:
     - `PartitionedParameterCoordinator`: Manages parameter fetch/release
     - `InflightParamRegistry`: Tracks parameters currently in use
   - **Key Methods**:
     - `fetch_sub_module()`: All-gather parameters before module execution
     - `release_sub_module()`: Release parameters after module execution

### Code Flow for Forward Pass

```python
# File: deepspeed/runtime/zero/partitioned_param_coordinator.py

class PartitionedParameterCoordinator:
    def fetch_sub_module(self, sub_module):
        """Called before each module's forward pass"""
        # 1. Get list of parameters in this module
        params = list(sub_module.parameters())

        # 2. All-gather parameters from all GPUs
        for param in params:
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                # Parameter is partitioned, need to gather
                all_gathered_param = self._all_gather_params(param)
                param.data = all_gathered_param
                param.ds_status = ZeroParamStatus.AVAILABLE

        # 3. Mark as inflight (in use)
        self.inflight_params.add(params)

    def release_sub_module(self, sub_module):
        """Called after each module's forward pass"""
        # 1. Get parameters
        params = list(sub_module.parameters())

        # 2. Release full parameters, keep only partition
        for param in params:
            if param.ds_status == ZeroParamStatus.AVAILABLE:
                # Free the gathered data
                param.data = param.ds_tensor  # Revert to partition
                param.ds_status = ZeroParamStatus.NOT_AVAILABLE

        # 3. Remove from inflight
        self.inflight_params.remove(params)

# File: deepspeed/runtime/zero/partition_parameters.py

def _all_gather_params(param):
    """Gather full parameter from all GPUs"""
    world_size = dist.get_world_size()

    # 1. Allocate buffer for full parameter
    full_param_buffer = torch.empty(
        param.ds_numel,  # Full parameter size
        dtype=param.dtype,
        device=param.device
    )

    # 2. All-gather operation
    dist.all_gather_into_tensor(
        full_param_buffer,  # Output: full parameter
        param.ds_tensor,    # Input: my partition
        group=param.ds_process_group
    )

    return full_param_buffer
```

### Hooks Installation

DeepSpeed installs PyTorch hooks to automatically trigger fetch/release:

```python
# File: deepspeed/runtime/zero/stage3.py

def _register_hooks(self, module):
    """Register forward and backward hooks for ZeRO-3"""

    # Forward hooks
    module.register_forward_pre_hook(self._pre_forward_hook)
    module.register_forward_hook(self._post_forward_hook)

    # Backward hooks
    module.register_full_backward_pre_hook(self._pre_backward_hook)
    module.register_full_backward_hook(self._post_backward_hook)

def _pre_forward_hook(self, module, inputs):
    """Called before module.forward()"""
    self.param_coordinator.fetch_sub_module(module)
    return inputs

def _post_forward_hook(self, module, inputs, outputs):
    """Called after module.forward()"""
    self.param_coordinator.release_sub_module(module)
    return outputs
```

### Detailed Example: One Layer's Journey

Let's trace a single layer through one forward-backward pass:

```python
# Initial state: Each GPU has 1/4 of layer.weight
# GPU 0: weight[0:250]
# GPU 1: weight[250:500]
# GPU 2: weight[500:750]
# GPU 3: weight[750:1000]

# ===== FORWARD PASS =====

# 1. Pre-forward hook triggered
_pre_forward_hook(layer, inputs)
├─ fetch_sub_module(layer)
├─ _all_gather_params(layer.weight)
│  ├─ GPU 0 sends weight[0:250] to all
│  ├─ GPU 1 sends weight[250:500] to all
│  ├─ GPU 2 sends weight[500:750] to all
│  └─ GPU 3 sends weight[750:1000] to all
└─ Now all GPUs have full weight[0:1000]

# 2. Forward computation
output = layer.forward(input, weight)
# Uses full weight[0:1000] on each GPU

# 3. Post-forward hook triggered
_post_forward_hook(layer, inputs, output)
├─ release_sub_module(layer)
└─ Each GPU reverts to its partition:
   ├─ GPU 0: weight[0:250]
   ├─ GPU 1: weight[250:500]
   ├─ GPU 2: weight[500:750]
   └─ GPU 3: weight[750:1000]

# ===== BACKWARD PASS =====

# 4. Pre-backward hook triggered
_pre_backward_hook(layer, grad_output)
├─ fetch_sub_module(layer) # Gather parameters again!
└─ Now all GPUs have full weight[0:1000] again

# 5. Backward computation
grad_input, grad_weight = layer.backward(grad_output, weight)
# Each GPU computes full grad_weight[0:1000]

# 6. Post-backward hook triggered
_post_backward_hook(layer, grad_output, grad_input)
├─ _reduce_scatter_gradients(grad_weight)
│  # Sum and partition gradients
│  ├─ GPU 0 gets sum(grad_weight[0:250])
│  ├─ GPU 1 gets sum(grad_weight[250:500])
│  ├─ GPU 2 gets sum(grad_weight[500:750])
│  └─ GPU 3 gets sum(grad_weight[750:1000])
├─ release_sub_module(layer)
└─ Each GPU reverts to its partition

# ===== OPTIMIZER STEP =====

# 7. Each GPU updates its partition independently
GPU 0: weight[0:250] -= lr * grad[0:250]
GPU 1: weight[250:500] -= lr * grad[250:500]
GPU 2: weight[500:750] -= lr * grad[500:750]
GPU 3: weight[750:1000] -= lr * grad[750:1000]

# No communication needed!
```

---

## Configuration Parameters

### Essential ZeRO-3 Config

```json
{
  "zero_optimization": {
    "stage": 3,

    // Prefetching
    "stage3_prefetch_bucket_size": 50000000,

    // Parameter persistence
    "stage3_param_persistence_threshold": 100000,

    // Maximum live parameters
    "stage3_max_live_parameters": 1000000000,

    // Communication optimization
    "overlap_comm": true,
    "contiguous_gradients": true,

    // Offloading
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

### Parameter Explanations

#### `stage3_prefetch_bucket_size`
- **What**: Number of parameters to prefetch ahead of time
- **Impact**: Larger = more memory, less latency
- **Default**: Auto-tuned by DeepSpeed
- **Typical**: 5e7 to 5e8 (50MB to 500MB)

#### `stage3_param_persistence_threshold`
- **What**: Parameters smaller than this stay persistent in GPU
- **Why**: Small parameters accessed frequently, gathering overhead too high
- **Default**: Auto-tuned
- **Typical**: 1e4 to 1e6 (10KB to 1MB)

#### `stage3_max_live_parameters`
- **What**: Maximum parameters in GPU simultaneously
- **Impact**: Lower = more memory savings, more communication
- **Default**: Auto-tuned
- **Typical**: 1e9 to 1e10 (1B to 10B params)

#### `overlap_comm`
- **What**: Overlap All-Gather with computation
- **Critical**: Always set to `true` for performance
- **Impact**: Can hide 50-80% of communication latency

---

## Performance Considerations

### Communication Volume

For a model with P parameters on N GPUs:

**Per Forward-Backward Pass**:
- All-Gather (forward): (N-1)/N × P × 2 bytes (per layer)
- All-Gather (backward): (N-1)/N × P × 2 bytes (per layer)
- Reduce-Scatter (backward): (N-1)/N × P × 2 bytes (per layer)
- **Total**: ~3 × P × 2 bytes per layer

**Example**: 7B parameter model, 8 GPUs, 32 transformer layers
- Per layer: ~7B/32 = 219M params = 438MB (BF16)
- All-Gather forward: 438MB × 7/8 = 383MB
- All-Gather backward: 383MB
- Reduce-Scatter: 383MB
- **Total per layer**: ~1.15GB
- **Total all layers**: 1.15GB × 32 = 36.8GB communication per step!

### Optimization Strategies

#### 1. **Overlap Communication with Computation**
```
Without overlap:
[Gather Layer 1] [Compute Layer 1] [Gather Layer 2] [Compute Layer 2] ...
        ↓ wasted time         ↓ wasted time

With overlap:
[Gather Layer 1] [Compute Layer 1] [Compute Layer 2] ...
                 [Gather Layer 2]  [Gather Layer 3]
                        ↑ overlapped!
```

Enable with: `"overlap_comm": true`

#### 2. **Gradient Accumulation**
Amortize communication cost over multiple micro-batches:

```
Without grad accum:
[Forward] [Backward+Comm] [Optimizer] × N times

With grad accum (N micro-batches):
[Forward] [Backward+Comm] × N
[Optimizer] × 1  ← Communication amortized!
```

#### 3. **Activation Checkpointing**
Trade computation for memory:

```
Normal: Store all activations → More memory
Checkpointing: Recompute activations → Less memory, more compute
```

Enables larger batch sizes, which amortizes communication cost better.

### When to Use ZeRO-3

✅ **Use ZeRO-3 when**:
- Model doesn't fit in single GPU memory
- Have multiple GPUs (more GPUs = better scaling)
- Training large models (>7B parameters)
- Have fast GPU interconnect (NVLink, InfiniBand)

❌ **Don't use ZeRO-3 when**:
- Model fits comfortably in single GPU
- Few GPUs (2-4) with slow interconnect
- Very small models (<1B parameters)
- Inference (use model parallelism instead)

### Comparison with Alternatives

| Method | Memory Savings | Communication | Complexity |
|--------|---------------|---------------|------------|
| Data Parallel | 1× | Low | Low |
| ZeRO-1 | 4× | Low | Low |
| ZeRO-2 | 8× | Medium | Medium |
| **ZeRO-3** | **N×** | **High** | **Medium** |
| Model Parallel | N× | Low | High |
| Pipeline Parallel | N× | Medium | High |

---

## Summary

**ZeRO-3 in One Sentence**:
Partition all model states (parameters, gradients, optimizer) across GPUs, gathering parameters on-demand during computation to achieve N× memory reduction.

**Key Mechanisms**:
1. **Partitioning**: Split parameters into N chunks, one per GPU
2. **All-Gather**: Reconstruct full parameters when needed
3. **Reduce-Scatter**: Sum and partition gradients
4. **Hooks**: Automatic fetch/release before/after each module

**Source Code Entry Points**:
1. `deepspeed/runtime/zero/stage3.py` - Main orchestration
2. `deepspeed/runtime/zero/partition_parameters.py` - Partitioning logic
3. `deepspeed/runtime/zero/partitioned_param_coordinator.py` - Fetch/release coordination

**When to Use**:
Training large models (>7B params) that don't fit in single GPU memory, with sufficient GPUs and fast interconnect.

---

*For practical examples, see the annotated scripts in `../annotated_scripts/`*
*For data flow details, see `Distributed_Training_Guide.md`*
