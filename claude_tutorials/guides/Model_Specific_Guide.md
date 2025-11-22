# Model-Specific DeepSpeed Configuration Guide

This guide explains how to use the production-ready DeepSpeed configurations provided for popular model architectures. Each configuration is optimized for specific model sizes, hardware setups, and training scenarios.

## Table of Contents

1. [Overview](#overview)
2. [LLaMA Models](#llama-models)
3. [GPT Models](#gpt-models)
4. [BERT Models](#bert-models)
5. [T5 Models](#t5-models)
6. [Configuration Selection Guide](#configuration-selection-guide)
7. [Customization Tips](#customization-tips)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### Available Configurations

We provide 13 production-ready configurations across 4 model families:

| Model Family | Configurations | Use Cases |
|--------------|----------------|-----------|
| **LLaMA** | 4 configs | Single-node training, CPU offload, multi-node scaling, LoRA fine-tuning |
| **GPT** | 3 configs | Baseline training, medium-scale, large-scale with offload |
| **BERT** | 2 configs | Fine-tuning, pre-training |
| **T5** | 4 configs | Small fine-tuning, base pre-training, large-scale, multi-node |

### Configuration Naming Convention

Configurations follow this pattern:
```
{model}_{size}_{optimization}.json
```

Examples:
- `llama_7b_single_node.json` - LLaMA 7B optimized for single node
- `llama_13b_zero3_offload.json` - LLaMA 13B with ZeRO-3 and CPU offload
- `gpt_neox_20b_zero3.json` - GPT-NeoX 20B with ZeRO-3

### Quick Start

```bash
# Using a configuration with deepspeed launcher
deepspeed --num_gpus=8 train.py \
  --deepspeed_config claude_tutorials/model_configs/llama/llama_7b_single_node.json

# Multi-node training
deepspeed --hostfile=hostfile train.py \
  --deepspeed_config claude_tutorials/model_configs/llama/llama_70b_multi_node.json
```

---

## LLaMA Models

### 1. LLaMA 7B Single Node (`llama_7b_single_node.json`)

**Hardware:** Single node with 8x A100 (40GB) or equivalent
**Memory:** ~28GB per GPU
**Training Speed:** ~2,500 tokens/sec/GPU

**Configuration Highlights:**
```json
{
  "train_batch_size": 128,
  "train_micro_batch_size_per_gpu": 4,
  "zero_optimization": {"stage": 2}
}
```

**Key Features:**
- ZeRO-2 for optimizer + gradient partitioning
- BF16 mixed precision for numerical stability
- Communication overlap enabled for performance
- Gradient accumulation: 4 steps

**When to Use:**
- Training LLaMA 7B from scratch
- Fine-tuning on instruction datasets
- Single-node setups with 8 GPUs
- Maximum throughput without offloading

**Modifications for Different Hardware:**
```python
# For 4x A100 (80GB):
"train_micro_batch_size_per_gpu": 8  # Double the batch size

# For 8x V100 (32GB):
"train_micro_batch_size_per_gpu": 2  # Reduce batch size
"gradient_accumulation_steps": 8      # Increase accumulation
```

**Expected Performance:**
- Throughput: ~20,000 tokens/sec (total)
- Memory usage: 28-32GB per GPU
- Training time (100B tokens): ~60 days

---

### 2. LLaMA 13B ZeRO-3 Offload (`llama_13b_zero3_offload.json`)

**Hardware:** Single node with 8x A100 (40GB) or 4x A100 (80GB)
**Memory:** GPU: 35GB, CPU RAM: 128GB+
**Training Speed:** ~1,200 tokens/sec/GPU

**Configuration Highlights:**
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

**Key Features:**
- ZeRO-3 partitions parameters, gradients, and optimizer states
- CPU optimizer offload to save GPU memory
- Activation checkpointing enabled
- BF16 mixed precision

**When to Use:**
- Training LLaMA 13B with limited GPU memory
- Single-node training without NVMe
- Cost optimization (smaller GPU clusters)
- Memory-constrained environments

**CPU Requirements:**
- Minimum: 128GB RAM
- Recommended: 256GB RAM for better performance
- Fast DDR4/DDR5 memory recommended

**Trade-offs:**
- **Pro:** Fits 13B model on 8x 40GB GPUs
- **Pro:** 30-40% cost savings vs larger GPUs
- **Con:** 20-30% slower than pure GPU training
- **Con:** CPU-GPU bandwidth bottleneck

**Performance Tuning:**
```json
// Increase overlap for better performance
"overlap_comm": true,
"sub_group_size": 1000000000,

// Tune prefetch for your hardware
"stage3_prefetch_bucket_size": 50000000,
"stage3_param_persistence_threshold": 100000
```

---

### 3. LLaMA 70B Multi-Node (`llama_70b_multi_node.json`)

**Hardware:** 4-8 nodes, 8x A100 (80GB) per node
**Memory:** 70-75GB per GPU
**Training Speed:** ~400 tokens/sec/GPU

**Configuration Highlights:**
```json
{
  "train_batch_size": 512,
  "zero_optimization": {"stage": 3},
  "optimizer": {
    "type": "OneBitAdam",
    "params": {"freeze_step": 2000}
  }
}
```

**Key Features:**
- ZeRO-3 for maximum parameter partitioning
- 1-bit Adam for 26x gradient communication compression
- Pipeline parallelism integration ready
- Activation checkpointing with CPU offload

**When to Use:**
- Pre-training LLaMA 70B from scratch
- Multi-node clusters (32-64 GPUs)
- High-throughput training at scale
- Research experiments requiring large models

**Network Requirements:**
- **Minimum:** 100 Gbps Ethernet with RoCE
- **Recommended:** InfiniBand HDR (200 Gbps)
- **Critical:** Low-latency interconnect (<10μs)

**Multi-Node Setup:**
```bash
# Create hostfile
cat > hostfile << EOF
node1 slots=8
node2 slots=8
node3 slots=8
node4 slots=8
EOF

# Launch training
deepspeed --hostfile=hostfile \
  --master_addr=node1 \
  --master_port=29500 \
  train.py \
  --deepspeed_config claude_tutorials/model_configs/llama/llama_70b_multi_node.json
```

**1-bit Adam Benefits:**
- Reduces gradient communication by 26x
- Essential for multi-node training
- Minimal accuracy impact after warmup
- Freeze step: 2000 (use FP32 Adam first for stability)

**Expected Performance:**
- Throughput: ~12,800 tokens/sec (32 GPUs)
- Memory usage: 72-78GB per GPU
- Training time (1T tokens): ~90 days on 32 GPUs

---

### 4. LLaMA LoRA Fine-tuning (`llama_lora_finetune.json`)

**Hardware:** Single node with 4-8 GPUs (A100/A6000)
**Memory:** ~18GB per GPU (7B), ~35GB per GPU (13B)
**Training Speed:** ~3,500 tokens/sec/GPU

**Configuration Highlights:**
```json
{
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 2,
  "zero_optimization": {"stage": 2},
  "optimizer": {
    "params": {"lr": 3e-4}  // Higher LR for LoRA
  }
}
```

**Key Features:**
- Optimized for LoRA adapter training
- Higher learning rate (3e-4 vs 2e-5 for full fine-tuning)
- Smaller gradient accumulation (faster updates)
- ZeRO-2 sufficient for adapter parameters

**When to Use:**
- Fine-tuning LLaMA with LoRA/QLoRA
- Task-specific adaptation
- Limited compute budget
- Rapid experimentation

**LoRA-Specific Considerations:**
```python
# Typical LoRA setup
lora_config = {
    "r": 8,              # Rank
    "lora_alpha": 16,    # Scaling factor
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.05
}

# Only ~0.5% of parameters are trainable
# 7B model: ~35M trainable params
# 13B model: ~65M trainable params
```

**Advantages:**
- 10x faster training than full fine-tuning
- 90% memory savings
- Easy to merge adapters later
- Multiple adapters for different tasks

**Training Script Integration:**
```python
from peft import LoraConfig, get_peft_model

# Create LoRA model
lora_config = LoraConfig(r=8, lora_alpha=16, ...)
model = get_peft_model(base_model, lora_config)

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config="claude_tutorials/model_configs/llama/llama_lora_finetune.json"
)
```

---

## GPT Models

### 1. GPT-2 Baseline (`gpt2_baseline.json`)

**Hardware:** Single node with 1-4 GPUs
**Memory:** ~8GB per GPU
**Training Speed:** ~8,000 tokens/sec/GPU

**Configuration Highlights:**
```json
{
  "train_batch_size": 256,
  "train_micro_batch_size_per_gpu": 16,
  "zero_optimization": {"stage": 1}
}
```

**Key Features:**
- ZeRO-1 (optimizer partitioning only)
- FP16 mixed precision
- Large batch size for stable training
- Minimal overhead configuration

**When to Use:**
- Training GPT-2 (124M-355M params)
- Educational purposes
- Baseline experiments
- Small-scale language modeling

**Model Variants:**
- GPT-2 Small (124M): 4-6GB per GPU
- GPT-2 Medium (355M): 6-8GB per GPU
- GPT-2 Large (774M): 12-16GB per GPU

**Performance Characteristics:**
- Fastest training speed per parameter
- Minimal memory overhead
- Good for rapid iteration
- Baseline for larger models

---

### 2. GPT-J 6B ZeRO-2 (`gptj_6b_zero2.json`)

**Hardware:** Single node with 8x A100 (40GB)
**Memory:** ~32GB per GPU
**Training Speed:** ~1,800 tokens/sec/GPU

**Configuration Highlights:**
```json
{
  "train_batch_size": 128,
  "train_micro_batch_size_per_gpu": 4,
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true
  }
}
```

**Key Features:**
- ZeRO-2 for gradient + optimizer partitioning
- BF16 for better numerical stability
- Communication overlap
- Optimized for A100 tensor cores

**When to Use:**
- Training GPT-J 6B
- Medium-scale language models
- Single-node setups
- Production fine-tuning

**Attention Mechanism:**
- Rotary Position Embeddings (RoPE)
- Parallel attention/FFN (unique to GPT-J)
- Requires careful learning rate tuning

**Hyperparameter Recommendations:**
```json
{
  "lr": 1.2e-4,           // Lower than GPT-2
  "warmup_steps": 2000,   // Longer warmup
  "weight_decay": 0.1     // Standard value
}
```

---

### 3. GPT-NeoX 20B ZeRO-3 (`gpt_neox_20b_zero3.json`)

**Hardware:** Single node with 8x A100 (80GB) or multi-node
**Memory:** GPU: 70GB per GPU, CPU: 256GB RAM
**Training Speed:** ~600 tokens/sec/GPU

**Configuration Highlights:**
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"}
  }
}
```

**Key Features:**
- ZeRO-3 with full CPU offloading
- Parameters and optimizer states on CPU
- Activation checkpointing with CPU offload
- NVMe offload ready

**When to Use:**
- Training GPT-NeoX 20B
- Large-scale models on limited hardware
- Cost-sensitive training
- Research on 20B-scale models

**Offloading Strategy:**
```
GPU: Activations + Computation
CPU: Parameters + Optimizer States
NVMe (optional): Overflow storage
```

**Performance vs Memory Trade-off:**
```
No Offload:    8x A100 (80GB) = $24/hr  → 1,200 tok/sec/GPU
CPU Offload:   8x A100 (40GB) = $16/hr  →   600 tok/sec/GPU
NVMe Offload:  8x A100 (40GB) = $16/hr  →   300 tok/sec/GPU

Cost savings: 33% | Speed reduction: 50-75%
```

**When to Add NVMe Offload:**
```json
"offload_param": {
  "device": "nvme",
  "nvme_path": "/local_nvme",
  "buffer_count": 5,
  "buffer_size": 500000000
}
```

Use NVMe when:
- CPU RAM < 256GB
- Training models >20B parameters
- Cost is more important than speed

---

## BERT Models

### 1. BERT Base Fine-tuning (`bert_base_finetuning.json`)

**Hardware:** Single GPU (T4/V100/A100)
**Memory:** ~6GB per GPU
**Training Speed:** ~200 samples/sec/GPU

**Configuration Highlights:**
```json
{
  "train_batch_size": 128,
  "train_micro_batch_size_per_gpu": 32,
  "zero_optimization": {"stage": 0}
}
```

**Key Features:**
- ZeRO-0 (disabled - model fits on single GPU)
- FP16 mixed precision
- Large batch size for classification tasks
- Minimal overhead for small models

**When to Use:**
- Fine-tuning BERT Base (110M params)
- Text classification, NER, QA tasks
- Single-GPU training
- Quick experiments

**Task-Specific Batch Sizes:**
```python
# Sequence Classification (GLUE, sentiment)
"train_micro_batch_size_per_gpu": 32

# Token Classification (NER)
"train_micro_batch_size_per_gpu": 16

# Question Answering (SQuAD)
"train_micro_batch_size_per_gpu": 12

# Long sequences (512 tokens)
"train_micro_batch_size_per_gpu": 8
```

**Learning Rate Guidelines:**
- Classification: 2e-5 to 5e-5
- NER: 3e-5 to 5e-5
- QA: 3e-5
- Warmup: 10% of total steps

---

### 2. BERT Large Pre-training (`bert_large_pretraining.json`)

**Hardware:** Multi-node with 32-64 GPUs
**Memory:** ~24GB per GPU
**Training Speed:** ~800 samples/sec/GPU

**Configuration Highlights:**
```json
{
  "train_batch_size": 2048,
  "zero_optimization": {"stage": 2},
  "optimizer": {
    "type": "OneBitLamb"
  }
}
```

**Key Features:**
- ZeRO-2 for distributed training
- 1-bit LAMB optimizer (memory + communication efficient)
- Massive batch size (2048)
- Optimized for pre-training from scratch

**When to Use:**
- Pre-training BERT Large (340M params)
- Domain-specific BERT models
- Multi-node clusters
- Large-scale pre-training

**1-bit LAMB Benefits:**
- 16x communication reduction
- Enables large batch training
- Better convergence than Adam for BERT
- Designed for BERT pre-training

**Pre-training Phases:**

**Phase 1 (Sequence Length 128):**
```json
{
  "train_batch_size": 2048,
  "sequence_length": 128,
  "steps": 90000
}
```

**Phase 2 (Sequence Length 512):**
```json
{
  "train_batch_size": 512,
  "sequence_length": 512,
  "steps": 10000
}
```

**Dataset Requirements:**
- Minimum: 100GB text (Wikipedia + Books)
- Recommended: 1TB+ for production models
- Preprocessing: WordPiece tokenization

---

## T5 Models

### 1. T5 Small Fine-tuning (`t5_small_finetuning.json`)

**Hardware:** Single node with 2-4 GPUs
**Memory:** ~10GB per GPU
**Training Speed:** ~150 samples/sec/GPU

**Configuration Highlights:**
```json
{
  "train_batch_size": 64,
  "train_micro_batch_size_per_gpu": 16,
  "zero_optimization": {"stage": 0}
}
```

**Key Features:**
- ZeRO-0 (model fits easily)
- FP16 mixed precision
- Optimized for seq2seq tasks
- Fast iteration for experiments

**When to Use:**
- Fine-tuning T5 Small (60M params)
- Translation, summarization, QA
- Limited compute budget
- Rapid prototyping

**T5 Task Formats:**
```python
# Translation
"translate English to German: The house is wonderful."

# Summarization
"summarize: [long article text...]"

# Question Answering
"question: What is the capital? context: [passage...]"
```

**Sequence Length Considerations:**
```json
// Short tasks (translation, classification)
"max_source_length": 512,
"max_target_length": 128

// Long tasks (summarization)
"max_source_length": 1024,
"max_target_length": 256
```

---

### 2. T5 Base Pre-training (`t5_base_pretraining.json`)

**Hardware:** Single node with 8x A100 (40GB)
**Memory:** ~28GB per GPU
**Training Speed:** ~350 samples/sec/GPU

**Configuration Highlights:**
```json
{
  "train_batch_size": 256,
  "zero_optimization": {"stage": 2},
  "activation_checkpointing": {
    "partition_activations": true
  }
}
```

**Key Features:**
- ZeRO-2 for efficient training
- Activation checkpointing enabled
- BF16 mixed precision
- Large batch size for stability

**When to Use:**
- Pre-training T5 Base (220M params)
- Custom T5 models for specific domains
- Multi-task learning
- Transfer learning research

**Pre-training Objectives:**

**C4 Dataset (Colossal Clean Crawled Corpus):**
- 750GB of cleaned web text
- Pre-processed with span corruption
- Typical training: 1M steps

**Span Corruption Example:**
```
Input:  "The cat sat on the <extra_id_0> mat and <extra_id_1>."
Target: "<extra_id_0> green <extra_id_1> purred"
```

**Training Schedule:**
```json
{
  "warmup_steps": 10000,
  "total_steps": 500000,
  "lr": 1e-4,
  "lr_schedule": "inverse_sqrt"
}
```

---

### 3. T5 Large ZeRO-3 (`t5_large_zero3.json`)

**Hardware:** Single node with 8x A100 (40GB/80GB)
**Memory:** GPU: 35GB, CPU: 128GB
**Training Speed:** ~180 samples/sec/GPU

**Configuration Highlights:**
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"}
  }
}
```

**Key Features:**
- ZeRO-3 with CPU offloading
- Activation checkpointing with CPU
- Small micro-batch size (2)
- High gradient accumulation

**When to Use:**
- Training T5 Large (770M params)
- Memory-constrained environments
- Cost optimization
- Single-node large model training

**Memory Breakdown:**
```
Model Parameters:     770M × 2 bytes (BF16) = 1.5GB
Activations:          ~20GB (batch size 2)
Optimizer (CPU):      770M × 12 bytes = 9.2GB
Gradients (CPU):      770M × 2 bytes = 1.5GB
```

**Tuning for Your Hardware:**
```python
# More GPU memory available (80GB):
"train_micro_batch_size_per_gpu": 4
"gradient_accumulation_steps": 4

# Less GPU memory (40GB):
"train_micro_batch_size_per_gpu": 1
"gradient_accumulation_steps": 16
"activation_checkpointing": {"cpu_checkpointing": true}
```

---

### 4. T5 XL Multi-Node (`t5_xl_multi_node.json`)

**Hardware:** 2-4 nodes with 8x A100 (80GB) each
**Memory:** GPU: 75GB, CPU: 256GB
**Training Speed:** ~80 samples/sec/GPU

**Configuration Highlights:**
```json
{
  "train_batch_size": 512,
  "zero_optimization": {"stage": 3},
  "optimizer": {"type": "OneBitAdam"}
}
```

**Key Features:**
- ZeRO-3 for maximum partitioning
- 1-bit Adam for communication efficiency
- Full CPU offloading
- Multi-node optimizations

**When to Use:**
- Pre-training T5 XL (3B params)
- Multi-node clusters (16-32 GPUs)
- Production-scale seq2seq models
- Large-scale multi-task learning

**Multi-Node Launch:**
```bash
# 4 nodes × 8 GPUs = 32 GPUs
deepspeed --num_nodes=4 \
  --num_gpus=8 \
  --master_addr=192.168.1.1 \
  --master_port=29500 \
  --hostfile=hostfile \
  train.py \
  --deepspeed_config claude_tutorials/model_configs/t5/t5_xl_multi_node.json
```

**Expected Training Time (C4 dataset):**
- 32 GPUs: ~45 days (1M steps)
- 64 GPUs: ~23 days (1M steps)
- Cost: ~$50k-$100k depending on cloud provider

**Network Requirements:**
- InfiniBand HDR strongly recommended
- NCCL 2.10+ with GPUDirect RDMA
- Low-latency interconnect critical

---

## Configuration Selection Guide

### By Model Size

| Parameters | ZeRO Stage | Offload | Hardware | Config Example |
|------------|------------|---------|----------|----------------|
| <500M | 0-1 | None | 1-4 GPUs | `gpt2_baseline.json` |
| 500M-3B | 2 | None | 4-8 GPUs | `gptj_6b_zero2.json` |
| 3B-13B | 2-3 | CPU | 8 GPUs | `llama_13b_zero3_offload.json` |
| 13B-30B | 3 | CPU | 8-16 GPUs | `t5_large_zero3.json` |
| 30B-100B | 3 | CPU+NVMe | 16-64 GPUs | `llama_70b_multi_node.json` |

### By Hardware Constraints

**Single GPU (V100/A100):**
- Models: BERT Base, GPT-2, T5 Small
- Configs: `*_finetuning.json`

**Single Node (8x A100 40GB):**
- Models: LLaMA 7B, GPT-J 6B, T5 Base
- Configs: `*_single_node.json`, `*_zero2.json`

**Single Node (8x A100 80GB):**
- Models: LLaMA 13B, GPT-NeoX 20B, T5 Large
- Configs: `*_zero3.json` (with offload)

**Multi-Node (32-64 GPUs):**
- Models: LLaMA 70B, T5 XL
- Configs: `*_multi_node.json`

### By Training Objective

**Fine-tuning (adapting pre-trained models):**
- Higher learning rates
- Smaller batch sizes
- Shorter training
- Configs: `bert_base_finetuning.json`, `llama_lora_finetune.json`

**Pre-training (from scratch):**
- Lower learning rates
- Larger batch sizes
- Long training runs
- Configs: `t5_base_pretraining.json`, `bert_large_pretraining.json`

### By Budget

**Low Budget (<$100):**
- Use LoRA/QLoRA configs
- Single-node with CPU offload
- Spot instances
- Example: `llama_lora_finetune.json` on 4x A100 spot

**Medium Budget ($100-$1000):**
- Single-node full fine-tuning
- ZeRO-2 without offload
- On-demand instances
- Example: `llama_7b_single_node.json` on 8x A100

**High Budget (>$1000):**
- Multi-node pre-training
- ZeRO-3 for scale
- Reserved instances
- Example: `llama_70b_multi_node.json` on 32-64 GPUs

---

## Customization Tips

### Adjusting Batch Size

**Rule of thumb:** Maximize batch size without OOM

```python
# Calculate effective batch size
effective_batch_size = (
    train_micro_batch_size_per_gpu ×
    gradient_accumulation_steps ×
    num_gpus
)

# Target: 128-512 for most LLMs
# BERT: 256-2048
```

**Memory vs Throughput:**
```json
// High memory, high throughput
"train_micro_batch_size_per_gpu": 8,
"gradient_accumulation_steps": 2

// Low memory, lower throughput
"train_micro_batch_size_per_gpu": 1,
"gradient_accumulation_steps": 16
```

### Tuning Learning Rates

**Starting points by model family:**
```python
LEARNING_RATES = {
    "llama": 2e-5,      # Conservative
    "gpt": 1.2e-4,       # Moderate
    "bert": 5e-5,        # Aggressive for fine-tuning
    "t5": 1e-4,          # Standard for seq2seq
    "lora": 3e-4,        # Higher for adapters
}
```

**Warmup schedules:**
```json
// Short training (<10k steps)
"warmup_steps": 500

// Medium training (10k-100k steps)
"warmup_steps": 2000

// Long training (>100k steps)
"warmup_steps": 10000
```

### Optimizing Communication

**For multi-node training:**
```json
{
  "zero_optimization": {
    "overlap_comm": true,              // Overlap communication
    "contiguous_gradients": true,      // Reduce fragmentation
    "reduce_bucket_size": 200000000,   // Larger buckets for IB
    "allgather_bucket_size": 200000000
  }
}
```

**For slow networks (Ethernet):**
```json
{
  "zero_optimization": {
    "reduce_bucket_size": 50000000,    // Smaller buckets
    "allgather_bucket_size": 50000000
  },
  "optimizer": {
    "type": "OneBitAdam"               // Compress gradients
  }
}
```

### Memory Optimization Hierarchy

**Level 1 - No offload (fastest):**
```json
{"zero_optimization": {"stage": 2}}
```

**Level 2 - Optimizer offload:**
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"}
  }
}
```

**Level 3 - Full offload:**
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"}
  }
}
```

**Level 4 - NVMe offload (slowest):**
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {
      "device": "nvme",
      "nvme_path": "/local_nvme"
    }
  }
}
```

### Activation Checkpointing

**When to enable:**
- Model doesn't fit in GPU memory
- Willing to trade 20-30% speed for memory

**Configuration:**
```json
{
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,        // GPU checkpointing
    "contiguous_memory_optimization": true,
    "number_checkpoints": 4            // Tune this
  }
}
```

**Number of checkpoints:**
```python
# Formula: sqrt(num_layers)
num_checkpoints = {
    "bert-base": 3,      # 12 layers
    "bert-large": 4,     # 24 layers
    "llama-7b": 5,       # 32 layers
    "llama-70b": 8,      # 80 layers
}
```

---

## Troubleshooting

### Out of Memory (OOM)

**Symptom:** `CUDA out of memory` error

**Solutions (in order):**

1. **Reduce micro-batch size:**
```json
"train_micro_batch_size_per_gpu": 1  // Start here
```

2. **Enable activation checkpointing:**
```json
"activation_checkpointing": {
  "partition_activations": true
}
```

3. **Increase ZeRO stage:**
```json
"zero_optimization": {"stage": 3}  // From 2
```

4. **Enable CPU offload:**
```json
"offload_optimizer": {"device": "cpu"}
```

5. **Enable parameter offload:**
```json
"offload_param": {"device": "cpu"}
```

### Slow Training Speed

**Symptom:** Low samples/sec, GPU utilization <80%

**Diagnosis:**
```python
# Check GPU utilization
nvidia-smi dmon -i 0 -s u

# Check if CPU-bound
htop  # Look for 100% CPU cores

# Check if I/O bound
iotop -o  # Look for high disk I/O
```

**Solutions:**

1. **Increase batch size:**
```json
"train_micro_batch_size_per_gpu": 8  // From 4
```

2. **Disable unnecessary offload:**
```json
// If you have enough GPU memory
"zero_optimization": {"stage": 2}  // From 3
```

3. **Enable communication overlap:**
```json
"overlap_comm": true
```

4. **Use faster data loading:**
```python
# In training script
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,        # Increase workers
    pin_memory=True,      # Pin memory
    prefetch_factor=2     # Prefetch batches
)
```

### Convergence Issues

**Symptom:** Loss not decreasing, NaN losses

**Solutions:**

1. **Check learning rate:**
```json
"optimizer": {
  "params": {
    "lr": 2e-5  // Try lower (divide by 10)
  }
}
```

2. **Increase warmup:**
```json
"scheduler": {
  "params": {
    "warmup_steps": 2000  // From 500
  }
}
```

3. **Enable gradient clipping:**
```json
"gradient_clipping": 1.0
```

4. **Check for FP16 overflow:**
```json
"fp16": {
  "enabled": true,
  "loss_scale": 0,              // Dynamic scaling
  "initial_scale_power": 12     // Lower if overflow
}
```

5. **Switch to BF16 (if available):**
```json
"bf16": {"enabled": true}  // More stable than FP16
```

### Multi-Node Issues

**Symptom:** Hanging, slow inter-node communication

**Diagnosis:**
```bash
# Test network bandwidth
iperf3 -c node2 -t 10

# Test NCCL
python -m torch.distributed.run \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=node1 \
  test_nccl.py
```

**Solutions:**

1. **Enable NCCL optimizations:**
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0          # Enable InfiniBand
export NCCL_IB_HCA=mlx5_0:1       # Specify IB device
export NCCL_SOCKET_IFNAME=ib0     # Use IB interface
```

2. **Tune bucket sizes:**
```json
"reduce_bucket_size": 200000000,    // Larger for IB
"allgather_bucket_size": 200000000
```

3. **Enable 1-bit Adam:**
```json
"optimizer": {"type": "OneBitAdam"}
```

### Config Loading Errors

**Symptom:** `KeyError`, `ValueError` when loading config

**Common causes:**

1. **Incompatible DeepSpeed version:**
```bash
# Check version
pip show deepspeed

# Upgrade
pip install --upgrade deepspeed
```

2. **Missing optimizer type:**
```json
// Make sure optimizer type is valid
"optimizer": {
  "type": "AdamW",  // Not "Adam" or "adam"
  "params": {...}
}
```

3. **Invalid ZeRO stage:**
```json
"zero_optimization": {
  "stage": 3  // Must be 0, 1, 2, or 3
}
```

---

## Advanced Topics

### Combining Techniques

**Maximum memory efficiency:**
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "nvme"}
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true
  }
}
```

**Maximum throughput:**
```json
{
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true
  },
  "train_micro_batch_size_per_gpu": 16,  // Large batch
  "bf16": {"enabled": true}
}
```

### Custom Optimizers

**Using Adafactor (memory-efficient):**
```json
{
  "optimizer": {
    "type": "Adafactor",
    "params": {
      "lr": 1e-3,
      "scale_parameter": true,
      "relative_step": false,
      "warmup_init": false
    }
  }
}
```

**Using LAMB (for large batch):**
```json
{
  "optimizer": {
    "type": "OneBitLamb",
    "params": {
      "lr": 6e-3,
      "weight_decay": 0.01,
      "bias_correction": true
    }
  }
}
```

### Pipeline Parallelism

**For very large models (>100B):**
```json
{
  "pipeline": {
    "enabled": true,
    "num_stages": 4  // Split model into 4 pipeline stages
  },
  "zero_optimization": {
    "stage": 1  // Use ZeRO-1 with pipeline
  }
}
```

---

## Summary

### Quick Reference

| Model | Size | Config | GPUs | Memory/GPU | Speed |
|-------|------|--------|------|------------|-------|
| BERT Base | 110M | `bert_base_finetuning.json` | 1 | 6GB | 200 samples/s |
| GPT-2 | 124M | `gpt2_baseline.json` | 1-4 | 8GB | 8000 tok/s |
| T5 Small | 60M | `t5_small_finetuning.json` | 2-4 | 10GB | 150 samples/s |
| GPT-J | 6B | `gptj_6b_zero2.json` | 8 | 32GB | 1800 tok/s |
| LLaMA 7B | 7B | `llama_7b_single_node.json` | 8 | 28GB | 2500 tok/s |
| T5 Base | 220M | `t5_base_pretraining.json` | 8 | 28GB | 350 samples/s |
| LLaMA 13B | 13B | `llama_13b_zero3_offload.json` | 8 | 35GB | 1200 tok/s |
| GPT-NeoX | 20B | `gpt_neox_20b_zero3.json` | 8 | 70GB | 600 tok/s |
| T5 Large | 770M | `t5_large_zero3.json` | 8 | 35GB | 180 samples/s |
| BERT Large | 340M | `bert_large_pretraining.json` | 32-64 | 24GB | 800 samples/s |
| LLaMA 70B | 70B | `llama_70b_multi_node.json` | 32-64 | 75GB | 400 tok/s |
| T5 XL | 3B | `t5_xl_multi_node.json` | 16-32 | 75GB | 80 samples/s |

### Next Steps

1. **Choose a configuration** based on your model and hardware
2. **Test with small dataset** to verify it works
3. **Monitor GPU utilization** and adjust batch size
4. **Profile memory usage** and enable offload if needed
5. **Scale to full dataset** and multi-node if required

### Additional Resources

- [Cost Optimization Guide](Cost_Optimization.md)
- [Multi-Node Setup Guide](Multi_Node_Setup.md)
- [Troubleshooting Guide](Troubleshooting_Guide.md)
- [DeepSpeed Documentation](https://deepspeed.readthedocs.io/)

---

**Configuration Repository:**
All configurations are in `claude_tutorials/model_configs/`

**Support:**
For issues, consult the Troubleshooting Guide or DeepSpeed GitHub Issues.
