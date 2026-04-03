# DeepSpeed Inference Optimization Tutorial

A comprehensive guide to optimizing model inference with DeepSpeed-Inference, covering kernel injection, quantization, tensor parallelism, and serving strategies.

---

## Table of Contents

1. [Introduction to DeepSpeed-Inference](#introduction-to-deepspeed-inference)
2. [Why Use DeepSpeed for Inference?](#why-use-deepspeed-for-inference)
3. [Kernel Injection](#kernel-injection)
4. [Quantization](#quantization)
5. [Tensor Parallelism for Inference](#tensor-parallelism-for-inference)
6. [ZeRO-Inference](#zero-inference)
7. [Configuration Guide](#configuration-guide)
8. [Performance Optimization](#performance-optimization)
9. [Production Deployment](#production-deployment)
10. [Troubleshooting](#troubleshooting)

---

## Introduction to DeepSpeed-Inference

### What is DeepSpeed-Inference?

**DeepSpeed-Inference** is a high-performance inference engine that accelerates transformer model inference through:
- Custom CUDA kernels
- Kernel fusion
- Quantization (INT8, FP16)
- Tensor parallelism
- Memory optimization

### Inference vs Training

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Goal** | Learn parameters | Generate predictions |
| **Batch Size** | Large (32-512) | Small (1-32) |
| **Latency** | Less critical | Critical |
| **Throughput** | Important | Critical |
| **Memory** | Gradients + optimizer | Model only |
| **Optimization** | Large batches, mixed precision | Low latency, high throughput |

---

## Why Use DeepSpeed for Inference?

### Performance Gains

| Model | Standard PyTorch | DeepSpeed-Inference | Speedup |
|-------|------------------|---------------------|---------|
| **GPT-2 (1.5B)** | 45 ms/token | 12 ms/token | 3.8× |
| **GPT-3 (6.7B)** | OOM | 38 ms/token | N/A (enables inference) |
| **BERT-Large** | 18 ms | 4 ms | 4.5× |
| **T5-3B** | 95 ms | 22 ms | 4.3× |

### Key Features

1. **Kernel Injection**: Replace PyTorch ops with optimized CUDA kernels
2. **Quantization**: INT8/FP16 for reduced memory and faster compute
3. **Tensor Parallelism**: Distribute large models across GPUs
4. **Model Compression**: Reduce model size without quality loss
5. **Automatic Optimization**: Detects and optimizes model architecture

---

## Kernel Injection

### What is Kernel Injection?

**Kernel injection** replaces PyTorch's default operations with highly optimized DeepSpeed kernels during model loading.

**Optimizations**:
- Fused attention (Flash Attention)
- Fused LayerNorm + residual
- Fused GELU activation
- Optimized matrix multiplication
- Reduced memory transfers

### How Kernel Injection Works

```
PyTorch Model (eager mode):
  LayerNorm → Dropout → Attention → Add → LayerNorm → FFN

DeepSpeed Injected Model:
  FusedLayerNormResidual → OptimizedAttention → FusedFFN
  (3 kernels instead of 10+)
```

**Result**: Fewer kernel launches, better memory access patterns.

---

### Enabling Kernel Injection

#### Basic Example

```python
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Initialize DeepSpeed-Inference
model = deepspeed.init_inference(
    model,
    mp_size=1,  # Tensor parallelism degree
    dtype=torch.float16,  # FP16 inference
    replace_with_kernel_inject=True,  # Enable kernel injection
)

# Inference
input_ids = tokenizer("Hello, world!", return_tensors="pt").input_ids.cuda()
output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0]))
```

---

### Supported Models

DeepSpeed kernel injection works with:

| Model Family | Support | Notes |
|--------------|---------|-------|
| **GPT** (GPT-2, GPT-J, GPT-NeoX) | ✅ Full | Best optimizations |
| **BERT** | ✅ Full | Encoder models |
| **T5** | ✅ Full | Encoder-decoder |
| **OPT** | ✅ Full | Meta's models |
| **BLOOM** | ✅ Full | BigScience models |
| **LLaMA** | ✅ Full | Excellent support |
| **Falcon** | ✅ Full | Latest architectures |
| **Custom** | ⚠️ Partial | May need manual config |

---

## Quantization

### Types of Quantization

#### 1. FP16 (Half Precision)
- **Bits**: 16
- **Speedup**: 2-3×
- **Quality**: Nearly lossless
- **Memory**: 2× reduction

```python
model = deepspeed.init_inference(
    model,
    dtype=torch.float16,
    replace_with_kernel_inject=True
)
```

#### 2. INT8 (8-bit Integer)
- **Bits**: 8
- **Speedup**: 3-4×
- **Quality**: Minor loss (< 1%)
- **Memory**: 4× reduction

```python
model = deepspeed.init_inference(
    model,
    dtype=torch.int8,
    quantization_setting=QuantizationConfig(
        q_bits=8,
        q_type=QuantizationType.ASYMMETRIC
    ),
    replace_with_kernel_inject=True
)
```

#### 3. Mixed Precision
- **Approach**: INT8 for weights, FP16 for activations
- **Speedup**: 2.5-3.5×
- **Quality**: Minimal loss
- **Memory**: 3× reduction

---

### Quantization-Aware Loading

DeepSpeed can quantize during model loading:

```python
import deepspeed
from deepspeed.ops.transformer.inference import QuantizationConfig

# Quantization config
quant_config = QuantizationConfig(
    q_bits=8,  # 8-bit quantization
    q_type=QuantizationType.ASYMMETRIC,  # Asymmetric for better accuracy
    q_groups=1,  # Group size for quantization
)

# Load and quantize
model = deepspeed.init_inference(
    model,
    dtype=torch.int8,
    quantization_setting=quant_config,
    replace_with_kernel_inject=True
)
```

---

### Post-Training Quantization (PTQ)

Quantize pre-trained model without retraining:

```python
from deepspeed.ops.transformer.inference import quantize_transformer

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Quantize
quantized_model = quantize_transformer(
    model,
    quant_bits=8,
    quant_type='asymmetric',
    quant_groups=1
)

# Use quantized model
quantized_model = deepspeed.init_inference(
    quantized_model,
    dtype=torch.int8,
    replace_with_kernel_inject=True
)
```

---

## Tensor Parallelism for Inference

### Why Tensor Parallelism?

**Problem**: Model too large for single GPU memory.

**Solution**: Split model across GPUs, each GPU holds a slice.

### How It Works

```
Single GPU:
  Linear(4096, 16384)  # 64M params, 128 MB (FP16)

Tensor Parallel (4 GPUs):
  GPU 0: Linear(4096, 4096)  # 16M params, 32 MB
  GPU 1: Linear(4096, 4096)  # 16M params, 32 MB
  GPU 2: Linear(4096, 4096)  # 16M params, 32 MB
  GPU 3: Linear(4096, 4096)  # 16M params, 32 MB
  Result: Concatenate outputs
```

---

### Enabling Tensor Parallelism

```python
import deepspeed

# Load model
model = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b")

# Initialize with tensor parallelism
model = deepspeed.init_inference(
    model,
    mp_size=4,  # Split across 4 GPUs
    dtype=torch.float16,
    replace_with_kernel_inject=True,
    max_out_tokens=512,
)

# Inference (automatically distributed)
output = model.generate(input_ids, max_length=100)
```

**Note**: Each GPU processes same batch, but holds different model slices.

---

### Choosing Tensor Parallelism Degree

| Model Size | Single GPU (A100 80GB) | Recommended mp_size |
|------------|------------------------|---------------------|
| < 7B params | ✅ Fits | 1 (no TP needed) |
| 7B-13B | ⚠️ Tight | 2 |
| 13B-30B | ❌ OOM | 4 |
| 30B-70B | ❌ OOM | 8 |
| 70B-175B | ❌ OOM | 16 |

**Formula**: `mp_size = ceil(model_size_gb / gpu_memory_gb)`

---

## ZeRO-Inference

### What is ZeRO-Inference?

**ZeRO-Inference** applies ZeRO memory optimization to inference:
- Partition model weights across GPUs
- Load weights on-demand during forward pass
- Minimize GPU memory for massive models

### When to Use ZeRO-Inference

✅ **Use if**:
- Model doesn't fit in GPU memory
- Latency is less critical (adds communication overhead)
- Running on many GPUs (8+)

❌ **Skip if**:
- Model fits in GPU
- Low latency is critical
- Single or few GPUs

---

### Enabling ZeRO-Inference

```python
import deepspeed

# Load model
model = AutoModelForCausalLM.from_pretrained("facebook/opt-30b")

# ZeRO-Inference config
ds_config = {
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",  # Offload to CPU if needed
            "pin_memory": True
        }
    }
}

# Initialize for inference
model = deepspeed.init_inference(
    model,
    config=ds_config,
    mp_size=8,  # Tensor parallelism
)

# Inference
output = model.generate(input_ids, max_length=100)
```

---

## Configuration Guide

### Minimal Configuration (Single GPU)

```python
import deepspeed
import torch

model = deepspeed.init_inference(
    model,
    dtype=torch.float16,  # FP16 for 2× speedup
    replace_with_kernel_inject=True,  # Use optimized kernels
)
```

---

### Optimized Configuration (Multi-GPU)

```python
model = deepspeed.init_inference(
    model,
    mp_size=4,  # Tensor parallelism across 4 GPUs
    dtype=torch.float16,
    replace_with_kernel_inject=True,
    replace_method='auto',  # Auto-detect model architecture
    max_out_tokens=512,  # Max generation length
    min_out_tokens=1,
)
```

---

### Production Configuration (Large Model)

```python
from deepspeed.ops.transformer.inference import Config

inference_config = Config(
    max_out_tokens=1024,
    min_out_tokens=1,
    mp_size=8,  # 8-way tensor parallelism
    replace_with_kernel_inject=True,
    dtype=torch.float16,
    enable_cuda_graph=False,  # Enable if using fixed input shapes
)

model = deepspeed.init_inference(
    model,
    config=inference_config
)
```

---

### Quantized Inference Configuration

```python
from deepspeed.ops.transformer.inference import QuantizationConfig, QuantizationType

quant_config = QuantizationConfig(
    q_bits=8,
    q_type=QuantizationType.ASYMMETRIC,
    q_groups=1
)

model = deepspeed.init_inference(
    model,
    dtype=torch.int8,
    quantization_setting=quant_config,
    replace_with_kernel_inject=True,
    mp_size=2
)
```

---

## Performance Optimization

### 1. Use Kernel Injection

**Impact**: 2-4× speedup

```python
# SLOW: Standard PyTorch
output = model(input_ids)

# FAST: DeepSpeed kernel injection
model = deepspeed.init_inference(
    model,
    replace_with_kernel_inject=True  # ← Critical!
)
output = model(input_ids)
```

---

### 2. Enable FP16

**Impact**: 2-3× speedup, 2× memory reduction

```python
model = deepspeed.init_inference(
    model,
    dtype=torch.float16  # ← FP16 inference
)
```

---

### 3. Use Larger Batch Sizes

**Impact**: Better GPU utilization

```python
# Process multiple sequences in parallel
input_ids = tokenizer(
    ["Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4"],
    return_tensors="pt",
    padding=True
).input_ids.cuda()

# Batch generation
outputs = model.generate(input_ids, max_length=50)
```

---

### 4. Optimize Generation Parameters

```python
# Faster generation with optimized sampling
output = model.generate(
    input_ids,
    max_length=50,
    num_beams=1,  # Greedy search (fastest)
    do_sample=False,  # No sampling overhead
    use_cache=True,  # Cache key/value (critical!)
    pad_token_id=tokenizer.eos_token_id
)
```

---

### 5. Use CUDA Graphs (Advanced)

For fixed input shapes:

```python
model = deepspeed.init_inference(
    model,
    dtype=torch.float16,
    replace_with_kernel_inject=True,
    enable_cuda_graph=True  # Reduce kernel launch overhead
)
```

**Requirements**:
- Fixed batch size
- Fixed sequence length
- No dynamic control flow

---

## Production Deployment

### Serving with FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# Load model once at startup
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = deepspeed.init_inference(
    model,
    dtype=torch.float16,
    replace_with_kernel_inject=True
)

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 50

@app.post("/generate")
def generate(request: GenerationRequest):
    input_ids = tokenizer(request.prompt, return_tensors="pt").input_ids.cuda()

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=request.max_length,
            do_sample=True,
            top_p=0.95,
            temperature=0.8
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"generated_text": text}

# Run: uvicorn server:app --host 0.0.0.0 --port 8000
```

---

### Load Balancing (Multiple GPUs)

```python
import torch.multiprocessing as mp
from queue import Queue

def worker(rank, model_name, request_queue, response_queue):
    """Worker process for one GPU."""
    torch.cuda.set_device(rank)

    # Load model on this GPU
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = deepspeed.init_inference(
        model,
        dtype=torch.float16,
        replace_with_kernel_inject=True
    )

    # Process requests
    while True:
        request = request_queue.get()
        if request is None:
            break

        input_ids, max_length = request
        output = model.generate(input_ids.cuda(), max_length=max_length)
        response_queue.put(output.cpu())

# Launch workers
num_gpus = 4
request_queue = mp.Queue()
response_queue = mp.Queue()

processes = []
for rank in range(num_gpus):
    p = mp.Process(target=worker, args=(rank, "gpt2", request_queue, response_queue))
    p.start()
    processes.append(p)

# Distribute requests
for prompt in prompts:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    request_queue.put((input_ids, 50))

# Collect responses
for _ in prompts:
    output = response_queue.get()
    print(tokenizer.decode(output[0]))

# Cleanup
for _ in range(num_gpus):
    request_queue.put(None)
for p in processes:
    p.join()
```

---

### Batching Requests

```python
import asyncio
from collections import deque

class BatchingEngine:
    def __init__(self, model, tokenizer, max_batch_size=8, max_wait_ms=50):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = deque()

    async def generate(self, prompt, max_length=50):
        """Add request to queue and wait for result."""
        future = asyncio.Future()
        self.queue.append((prompt, max_length, future))
        return await future

    async def process_batch(self):
        """Process batch of requests."""
        while True:
            # Wait for requests or timeout
            await asyncio.sleep(self.max_wait_ms / 1000)

            if not self.queue:
                continue

            # Collect batch
            batch = []
            futures = []
            while len(batch) < self.max_batch_size and self.queue:
                prompt, max_length, future = self.queue.popleft()
                batch.append(prompt)
                futures.append((future, max_length))

            # Process batch
            input_ids = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True
            ).input_ids.cuda()

            outputs = self.model.generate(input_ids, max_length=max_length)

            # Return results
            for i, (future, _) in enumerate(futures):
                text = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
                future.set_result(text)

# Usage
engine = BatchingEngine(model, tokenizer)
asyncio.create_task(engine.process_batch())

# Handle requests
result = await engine.generate("Hello, world!")
```

---

## Troubleshooting

### Issue 1: Kernel injection fails

**Error**:
```
WARNING: Kernel injection not supported for this model
```

**Solutions**:

#### Solution A: Check Model Compatibility
```python
from deepspeed.ops.transformer.inference import DeepSpeedTransformerInference

# Check if model is supported
print(DeepSpeedTransformerInference.supported_models)
```

#### Solution B: Manual Injection Config
```python
from deepspeed.ops.transformer.inference import Config

config = Config(
    hidden_size=768,
    heads=12,
    layer_norm_eps=1e-5,
    max_out_tokens=512
)

model = deepspeed.init_inference(
    model,
    config=config,
    replace_with_kernel_inject=True
)
```

---

### Issue 2: Out of Memory

**Error**:
```
RuntimeError: CUDA out of memory during inference
```

**Solutions**:

#### Solution A: Use Tensor Parallelism
```python
model = deepspeed.init_inference(
    model,
    mp_size=2,  # Split across 2 GPUs
    dtype=torch.float16
)
```

#### Solution B: Enable Quantization
```python
model = deepspeed.init_inference(
    model,
    dtype=torch.int8,  # 8-bit quantization
    replace_with_kernel_inject=True
)
```

#### Solution C: Reduce Batch Size
```python
# Process one sequence at a time
for prompt in prompts:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    output = model.generate(input_ids, max_length=50)
```

---

### Issue 3: Slow inference despite optimizations

**Problem**: Inference still slow after enabling DeepSpeed.

**Diagnosis**:

```python
import time

# Benchmark
start = time.time()
for _ in range(100):
    output = model.generate(input_ids, max_length=50)
torch.cuda.synchronize()
end = time.time()

avg_latency = (end - start) / 100
print(f"Average latency: {avg_latency*1000:.2f} ms")
```

**Solutions**:

#### Check if kernel injection actually enabled:
```python
# Look for this in model structure
print(model)
# Should see "DeepSpeedTransformerInference" layers
```

#### Ensure FP16:
```python
# Verify dtype
print(next(model.parameters()).dtype)  # Should be torch.float16
```

#### Profile:
```python
with torch.profiler.profile() as prof:
    output = model.generate(input_ids, max_length=50)

print(prof.key_averages().table())
```

---

## Best Practices Summary

1. **Always use kernel injection**: `replace_with_kernel_inject=True`
2. **Use FP16 by default**: `dtype=torch.float16`
3. **Enable caching**: `use_cache=True` in generation
4. **Batch requests**: Process multiple prompts together
5. **Use tensor parallelism** for models > 7B params
6. **Quantize for memory**: INT8 if memory constrained
7. **Benchmark**: Measure latency before deploying
8. **Warm-up**: Run a few inferences before measuring

---

## Complete Example: Optimized GPT-J Inference

```python
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def optimize_model_for_inference(model_name="EleutherAI/gpt-j-6B"):
    """Load and optimize model for inference."""
    print(f"Loading {model_name}...")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Optimize with DeepSpeed
    print("Initializing DeepSpeed-Inference...")
    model = deepspeed.init_inference(
        model,
        mp_size=2,  # Tensor parallelism across 2 GPUs
        dtype=torch.float16,
        replace_with_kernel_inject=True,
        max_out_tokens=512
    )

    print("Model optimized!")
    return model, tokenizer

def benchmark(model, tokenizer, prompts, max_length=50):
    """Benchmark inference performance."""
    print(f"\nBenchmarking with {len(prompts)} prompts...")

    # Warm-up
    for _ in range(3):
        input_ids = tokenizer(prompts[0], return_tensors="pt").input_ids.cuda()
        _ = model.generate(input_ids, max_length=max_length)

    # Benchmark
    start = time.time()
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        output = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
            use_cache=True
        )
    torch.cuda.synchronize()
    end = time.time()

    avg_latency = (end - start) / len(prompts)
    throughput = len(prompts) / (end - start)

    print(f"Average latency: {avg_latency*1000:.2f} ms/generation")
    print(f"Throughput: {throughput:.2f} generations/sec")

    return avg_latency

def main():
    # Optimize model
    model, tokenizer = optimize_model_for_inference()

    # Test prompts
    prompts = [
        "The future of artificial intelligence is",
        "In a world where technology has advanced",
        "The key to solving climate change is",
    ]

    # Benchmark
    latency = benchmark(model, tokenizer, prompts)

    # Generate examples
    print("\n" + "="*50)
    print("Example generations:")
    print("="*50)

    for prompt in prompts[:2]:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        output = model.generate(
            input_ids,
            max_length=100,
            do_sample=True,
            top_p=0.95,
            temperature=0.8
        )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {text}\n")

if __name__ == '__main__':
    main()
```

---

## Additional Resources

- **[DeepSpeed-Inference Documentation](https://www.deepspeed.ai/inference/)** - Official docs
- **[DeepSpeed-FastGen](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen)** - Latest inference optimizations
- **[Kernel Injection Guide](https://www.deepspeed.ai/tutorials/inference-tutorial/)** - Detailed tutorial
- **[Model Serving Best Practices](https://www.deepspeed.ai/tutorials/model-serving/)** - Production deployment

**Happy fast inference!** 🚀
