# DeepSpeed Custom Kernels Tutorial

A comprehensive guide to writing and integrating custom CUDA kernels with DeepSpeed using the OpBuilder system.

---

## Table of Contents

1. [Introduction to Custom Kernels](#introduction-to-custom-kernels)
2. [DeepSpeed OpBuilder System](#deepspeed-opbuilder-system)
3. [Writing Your First Kernel](#writing-your-first-kernel)
4. [Integrating with DeepSpeed](#integrating-with-deepspeed)
5. [Advanced Kernel Techniques](#advanced-kernel-techniques)
6. [Optimization Strategies](#optimization-strategies)
7. [Debugging Custom Kernels](#debugging-custom-kernels)
8. [Best Practices](#best-practices)

---

## Introduction to Custom Kernels

### Why Write Custom Kernels?

**Use cases**:
1. **Performance**: 5-10× speedup for specialized operations
2. **Memory efficiency**: Fused operations reduce memory transfers
3. **New operations**: Implement algorithms not in PyTorch
4. **Research**: Experiment with novel architectures

### CUDA Kernel Basics

A **CUDA kernel** is a function that runs on the GPU:

```cuda
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

**Key concepts**:
- `__global__`: Kernel function (callable from CPU)
- `blockIdx`, `threadIdx`: Thread coordinates
- Grid/block organization: Thousands of threads in parallel

---

## DeepSpeed OpBuilder System

### What is OpBuilder?

**OpBuilder** is DeepSpeed's system for compiling and loading custom CUDA ops at runtime.

**Benefits**:
- Just-in-time (JIT) compilation
- Automatic dependency management
- Easy integration with PyTorch
- Caching for fast reloads

### OpBuilder Architecture

```
Your Code:
  custom_op.cpp (C++ interface)
  custom_op_kernel.cu (CUDA implementation)
    ↓
OpBuilder:
  1. Compile .cpp and .cu files
  2. Link with PyTorch
  3. Create Python bindings
    ↓
PyTorch Module:
  import custom_op
  output = custom_op.forward(input)
```

---

## Writing Your First Kernel

### Example: Fused ReLU + LayerNorm

Let's implement a fused operation combining ReLU activation and LayerNorm.

#### Step 1: CUDA Kernel Implementation

**File**: `fused_relu_ln_kernel.cu`

```cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void fused_relu_ln_kernel(
    const T* input,
    T* output,
    const T* gamma,
    const T* beta,
    int batch_size,
    int hidden_size,
    float eps
) {
    int idx = blockIdx.x;  // Batch index
    int tid = threadIdx.x;  // Hidden dimension index

    // Step 1: ReLU
    extern __shared__ float shared[];
    float val = (float)input[idx * hidden_size + tid];
    val = val > 0.0f ? val : 0.0f;  // ReLU
    shared[tid] = val;
    __syncthreads();

    // Step 2: Compute mean
    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        sum += shared[i];
    }
    // Reduce across threads
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum += shared[tid + stride];
        }
        __syncthreads();
    }
    float mean = sum / hidden_size;
    __syncthreads();

    // Step 3: Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float diff = shared[i] - mean;
        var_sum += diff * diff;
    }
    // Reduce variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            var_sum += shared[tid + stride];
        }
        __syncthreads();
    }
    float variance = var_sum / hidden_size;
    float inv_std = rsqrtf(variance + eps);
    __syncthreads();

    // Step 4: Normalize and scale
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float normalized = (shared[i] - mean) * inv_std;
        float scaled = normalized * (float)gamma[i] + (float)beta[i];
        output[idx * hidden_size + i] = (T)scaled;
    }
}

// Launcher function
void fused_relu_ln_forward_cuda(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int batch_size,
    int hidden_size,
    float eps
) {
    dim3 grid(batch_size);
    dim3 block(min(hidden_size, 1024));
    size_t shared_mem = hidden_size * sizeof(float);

    fused_relu_ln_kernel<<<grid, block, shared_mem>>>(
        input, output, gamma, beta, batch_size, hidden_size, eps
    );
}
```

---

#### Step 2: C++ Interface

**File**: `fused_relu_ln.cpp`

```cpp
#include <torch/extension.h>

// Forward declaration of CUDA function
void fused_relu_ln_forward_cuda(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int batch_size,
    int hidden_size,
    float eps
);

// PyTorch interface
torch::Tensor fused_relu_ln_forward(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    // Check inputs
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "beta must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto batch_size = input.size(0);
    auto hidden_size = input.size(1);

    // Allocate output
    auto output = torch::empty_like(input);

    // Launch kernel
    fused_relu_ln_forward_cuda(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        batch_size,
        hidden_size,
        eps
    );

    return output;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_relu_ln_forward, "Fused ReLU + LayerNorm forward");
}
```

---

#### Step 3: OpBuilder Setup

**File**: `fused_relu_ln_builder.py`

```python
from deepspeed.ops.op_builder import OpBuilder

class FusedReLULNBuilder(OpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_RELU_LN"
    NAME = "fused_relu_ln"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f"deepspeed.ops.{self.NAME}_op"

    def sources(self):
        return [
            "fused_relu_ln.cpp",
            "fused_relu_ln_kernel.cu"
        ]

    def include_paths(self):
        return []

    def cxx_args(self):
        return ["-O3", "-std=c++14"]

    def nvcc_args(self):
        return [
            "-O3",
            "--use_fast_math",
            "-gencode", "arch=compute_70,code=sm_70",  # V100
            "-gencode", "arch=compute_80,code=sm_80",  # A100
        ]
```

---

#### Step 4: Build and Use

```python
import torch
from fused_relu_ln_builder import FusedReLULNBuilder

# Build the op
builder = FusedReLULNBuilder()
fused_relu_ln = builder.load()

# Use the kernel
batch_size = 32
hidden_size = 768

input = torch.randn(batch_size, hidden_size).cuda()
gamma = torch.ones(hidden_size).cuda()
beta = torch.zeros(hidden_size).cuda()

# Forward pass
output = fused_relu_ln.forward(input, gamma, beta, eps=1e-5)

print(f"Input shape: {input.shape}")
print(f"Output shape: {output.shape}")
```

---

## Integrating with DeepSpeed

### Using OpBuilder

DeepSpeed provides a base `OpBuilder` class for custom ops:

```python
from deepspeed.ops.op_builder import OpBuilder

class MyCustomOpBuilder(OpBuilder):
    BUILD_VAR = "DS_BUILD_MY_OP"  # Environment variable
    NAME = "my_custom_op"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        """Full Python module name."""
        return f"deepspeed.ops.{self.NAME}_op"

    def sources(self):
        """Source files to compile."""
        return ["my_op.cpp", "my_op_kernel.cu"]

    def include_paths(self):
        """Additional include directories."""
        return []

    def cxx_args(self):
        """C++ compiler flags."""
        return ["-O3", "-std=c++14", "-g"]

    def nvcc_args(self):
        """NVCC compiler flags."""
        return [
            "-O3",
            "--use_fast_math",
            "-gencode", "arch=compute_70,code=sm_70",
            "-gencode", "arch=compute_80,code=sm_80",
        ]

# Build and load
builder = MyCustomOpBuilder()
my_op = builder.load()
```

---

### PyTorch Module Wrapper

Wrap your custom op in a `nn.Module`:

```python
import torch
import torch.nn as nn
from my_custom_op_builder import MyCustomOpBuilder

class MyCustomLayer(nn.Module):
    """PyTorch layer using custom CUDA kernel."""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Build op
        builder = MyCustomOpBuilder()
        self.op = builder.load()

        # Parameters
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        return self.op.forward(x, self.weight, self.bias)

# Use in model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.custom_layer = MyCustomLayer(768)
        self.linear = nn.Linear(768, 10)

    def forward(self, x):
        x = self.custom_layer(x)
        x = self.linear(x)
        return x

model = MyModel().cuda()
```

---

## Advanced Kernel Techniques

### 1. Kernel Fusion

Combine multiple operations into single kernel:

```cuda
// Instead of:
//   1. ReLU kernel
//   2. LayerNorm kernel
//   3. Dropout kernel
// Do all in one kernel:

__global__ void fused_relu_ln_dropout_kernel(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    float dropout_prob,
    unsigned long long seed,
    int batch_size,
    int hidden_size
) {
    // 1. ReLU
    float val = input[...];
    val = val > 0.0f ? val : 0.0f;

    // 2. LayerNorm
    // ... (compute mean, variance, normalize)

    // 3. Dropout
    curandState state;
    curand_init(seed, idx, 0, &state);
    float random = curand_uniform(&state);
    val = (random > dropout_prob) ? val / (1 - dropout_prob) : 0.0f;

    output[...] = val;
}
```

**Benefits**: 3× fewer memory reads/writes.

---

### 2. Memory Coalescing

Ensure threads access contiguous memory:

```cuda
// BAD: Stride access (slow)
__global__ void bad_kernel(float* data, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    float val = data[col * cols + row];  // Non-coalesced!
}

// GOOD: Coalesced access (fast)
__global__ void good_kernel(float* data, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    float val = data[row * cols + col];  // Coalesced!
}
```

---

### 3. Shared Memory

Use shared memory for fast data sharing:

```cuda
__global__ void matrix_multiply_shared(
    float* A, float* B, float* C,
    int M, int N, int K
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < K / TILE_SIZE; ++t) {
        // Load tile into shared memory
        As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}
```

---

### 4. Warp-Level Primitives

Use warp shuffles for fast reductions:

```cuda
__device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void fast_reduction_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? input[idx] : 0.0f;

    // Warp-level reduction (very fast!)
    val = warp_reduce_sum(val);

    // Only first thread in warp writes
    if (threadIdx.x % warpSize == 0) {
        atomicAdd(output, val);
    }
}
```

---

### 5. Tensor Cores (FP16 Matmul)

Use Tensor Cores for matrix multiplication:

```cuda
#include <mma.h>
using namespace nvcuda;

__global__ void tensor_core_matmul(
    half* A, half* B, float* C,
    int M, int N, int K
) {
    // Tensor Core fragment
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0.0f);

    // Load fragments and compute
    for (int i = 0; i < K; i += 16) {
        wmma::load_matrix_sync(a_frag, A + ..., K);
        wmma::load_matrix_sync(b_frag, B + ..., K);

        // Matrix multiply-accumulate (on Tensor Cores!)
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result
    wmma::store_matrix_sync(C + ..., c_frag, N, wmma::mem_row_major);
}
```

---

## Optimization Strategies

### 1. Occupancy Optimization

**Goal**: Maximize active warps per SM.

**Tools**:
```bash
# Use CUDA Occupancy Calculator
nvcc --ptxas-options=-v my_kernel.cu

# Output shows:
# registers per thread: 32
# shared memory per block: 4096 bytes
# → Calculate optimal block size
```

**Guidelines**:
- Target 50-100% occupancy
- Balance registers, shared memory, block size
- Use `__launch_bounds__` to hint compiler

```cuda
__global__ void __launch_bounds__(256, 4)
my_kernel(...) {
    // Compiler optimizes for 256 threads/block, 4 blocks/SM
}
```

---

### 2. Instruction-Level Parallelism (ILP)

Unroll loops to increase ILP:

```cuda
// Low ILP
__global__ void low_ilp_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;  // One op at a time
    }
}

// High ILP (4× unroll)
__global__ void high_ilp_kernel(float* data, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        float val0 = data[idx + 0] * 2.0f;
        float val1 = data[idx + 1] * 2.0f;
        float val2 = data[idx + 2] * 2.0f;
        float val3 = data[idx + 3] * 2.0f;
        data[idx + 0] = val0;
        data[idx + 1] = val1;
        data[idx + 2] = val2;
        data[idx + 3] = val3;
    }
}
```

---

### 3. Minimize Divergence

Avoid branch divergence within warps:

```cuda
// BAD: High divergence
__global__ void divergent_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx % 2 == 0) {
        data[idx] = expensive_computation_a(data[idx]);
    } else {
        data[idx] = expensive_computation_b(data[idx]);
    }
    // Half the warp idle during each branch!
}

// GOOD: Separate warps for each case
__global__ void non_divergent_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Ensure all threads in warp take same path
    data[idx] = (idx < n/2) ?
        expensive_computation_a(data[idx]) :
        expensive_computation_b(data[idx]);
}
```

---

### 4. Use Streams for Overlap

Overlap kernel execution with data transfers:

```cpp
// C++ code
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Overlap transfers and compute
cudaMemcpyAsync(d_input1, h_input1, size, H2D, stream1);
cudaMemcpyAsync(d_input2, h_input2, size, H2D, stream2);

my_kernel<<<grid, block, 0, stream1>>>(d_input1, d_output1);
my_kernel<<<grid, block, 0, stream2>>>(d_input2, d_output2);

cudaMemcpyAsync(h_output1, d_output1, size, D2H, stream1);
cudaMemcpyAsync(h_output2, d_output2, size, D2H, stream2);
```

---

## Debugging Custom Kernels

### 1. CUDA Error Checking

Always check for errors:

```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Use:
CUDA_CHECK(cudaMalloc(&d_data, size));
my_kernel<<<grid, block>>>(d_data);
CUDA_CHECK(cudaDeviceSynchronize());
```

---

### 2. printf Debugging

Use printf in kernels:

```cuda
__global__ void debug_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {  // Only print from first thread
        printf("Block: %d, Thread: %d, Value: %f\n",
               blockIdx.x, threadIdx.x, data[idx]);
    }
}
```

---

### 3. CUDA-MEMCHECK

Check for memory errors:

```bash
# Run with cuda-memcheck
cuda-memcheck python train.py

# Output:
# ========= Invalid __global__ write of size 4
# =========     at 0x00000128 in my_kernel
# =========     by thread (0,0,0) in block (0,0,0)
```

---

### 4. NVIDIA Nsight

Profile kernels:

```bash
# Nsight Compute (kernel profiling)
ncu --set full --export profile python train.py

# Nsight Systems (timeline profiling)
nsys profile --trace=cuda,nvtx python train.py
```

---

### 5. Assert in Kernels

Use assert for bounds checking:

```cuda
__global__ void safe_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Assert bounds (only in debug builds)
    assert(idx < n && "Index out of bounds!");

    data[idx] = data[idx] * 2.0f;
}
```

---

## Best Practices

### 1. Start Simple, Then Optimize

```
1. Naive implementation (correctness)
2. Profile (identify bottlenecks)
3. Optimize hot spots
4. Benchmark (measure improvement)
5. Repeat
```

### 2. Use Existing Libraries When Possible

Before writing custom kernel, check:
- cuBLAS (matrix ops)
- cuDNN (conv, RNN, etc.)
- Thrust (algorithms)
- CUB (block-level primitives)
- DeepSpeed ops (fused kernels)

### 3. Benchmark Rigorously

```python
import torch
import time

def benchmark_kernel(kernel_func, input_data, num_iters=1000):
    # Warm-up
    for _ in range(10):
        output = kernel_func(input_data)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        output = kernel_func(input_data)
    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / num_iters
    print(f"Average time: {avg_time*1000:.3f} ms")
    return avg_time
```

### 4. Document Performance Characteristics

```python
class MyCustomOp:
    """
    Fused ReLU + LayerNorm kernel.

    Performance:
    - V100: 0.15 ms for (32, 768)
    - A100: 0.08 ms for (32, 768)
    - 3.2× faster than PyTorch (V100)
    - 4.1× faster than PyTorch (A100)

    Memory:
    - 2× reduction vs unfused (one read/write vs two)

    Limitations:
    - Requires contiguous tensors
    - Max hidden_size: 4096
    """
    pass
```

### 5. Version Your Kernels

```python
class MyOpBuilder(OpBuilder):
    VERSION = "1.2.0"  # Track versions

    def load(self):
        # Check cache with version
        cached_path = f"~/.cache/deepspeed/my_op_v{self.VERSION}.so"
        # ...
```

---

## Complete Example: Custom Softmax Kernel

```cuda
// softmax_kernel.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void softmax_kernel(
    const float* input,
    float* output,
    int batch_size,
    int seq_length
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];
    float* s_vals = shared;
    float* s_max = &shared[seq_length];

    // Load and find max
    float local_max = -CUDART_INF_F;
    for (int i = tid; i < seq_length; i += blockDim.x) {
        float val = input[batch_idx * seq_length + i];
        s_vals[i] = val;
        local_max = fmaxf(local_max, val);
    }

    // Reduce max across threads
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            local_max = fmaxf(local_max, s_max[tid + stride]);
        }
        __syncthreads();
    }

    // Broadcast max
    if (tid == 0) {
        s_max[0] = local_max;
    }
    __syncthreads();
    float max_val = s_max[0];

    // Exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < seq_length; i += blockDim.x) {
        float val = expf(s_vals[i] - max_val);
        s_vals[i] = val;
        local_sum += val;
    }

    // Reduce sum
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            local_sum += s_max[tid + stride];
        }
        __syncthreads();
    }

    // Broadcast sum
    if (tid == 0) {
        s_max[0] = local_sum;
    }
    __syncthreads();
    float sum = s_max[0];

    // Normalize
    for (int i = tid; i < seq_length; i += blockDim.x) {
        output[batch_idx * seq_length + i] = s_vals[i] / sum;
    }
}
```

```cpp
// softmax.cpp
#include <torch/extension.h>

void softmax_cuda_forward(
    const float* input,
    float* output,
    int batch_size,
    int seq_length
);

torch::Tensor softmax_forward(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto seq_length = input.size(1);
    auto output = torch::empty_like(input);

    softmax_cuda_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_length
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softmax_forward, "Custom Softmax forward");
}
```

---

## Additional Resources

- **[CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)** - Official CUDA docs
- **[DeepSpeed Op Builder](https://github.com/microsoft/DeepSpeed/tree/master/op_builder)** - OpBuilder source code
- **[PyTorch Custom Ops](https://pytorch.org/tutorials/advanced/cpp_extension.html)** - PyTorch extension tutorial
- **[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)** - Optimization guide
- **[Nsight Profiler](https://developer.nvidia.com/nsight-compute)** - Profiling tools

**Happy kernel development!** 🚀
