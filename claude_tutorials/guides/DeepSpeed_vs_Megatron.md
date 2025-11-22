# DeepSpeed vs Megatron-LM: Comprehensive Comparison

A detailed comparison between Microsoft DeepSpeed and NVIDIA Megatron-LM for training large language models at scale, plus coverage of their powerful integration: Megatron-DeepSpeed.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Feature Comparison](#feature-comparison)
4. [Megatron-DeepSpeed Integration](#megatron-deepspeed-integration)
5. [Performance Benchmarks](#performance-benchmarks)
6. [Code Examples](#code-examples)
7. [Use Case Recommendations](#use-case-recommendations)
8. [Advanced Topics](#advanced-topics)

---

## Executive Summary

### Quick Comparison

| Aspect | DeepSpeed | Megatron-LM | Megatron-DeepSpeed |
|--------|-----------|-------------|---------------------|
| **Maintainer** | Microsoft | NVIDIA | Microsoft + NVIDIA |
| **Primary Focus** | Memory optimization | Tensor parallelism | Best of both |
| **Key Technique** | ZeRO (data parallel) | Model parallel | 3D parallelism |
| **Best For** | Memory efficiency | Compute efficiency | Extreme scale (>100B) |
| **Model Support** | Any PyTorch model | GPT, BERT, T5 | GPT, BERT, T5 |
| **Ease of Use** | Easy | Moderate | Moderate |
| **FlexibilityUniversal training | GPU-specific optimization | Combined power |
| **Integration** | Minimal changes | Significant rewrite | Moderate changes |
| **Multi-Node Scaling** | Excellent | Excellent | Outstanding |

### Key Distinctions

**DeepSpeed** = **Data Parallelism** at Scale
- ZeRO shards optimizer states, gradients, parameters
- Each GPU has different data, same model (sharded)
- Communication: All-reduce/All-gather
- Focus: Fit larger models in memory

**Megatron-LM** = **Model Parallelism** at Scale
- Tensor parallelism splits model layers across GPUs
- Each GPU has same data, different model parts
- Communication: Point-to-point, all-reduce
- Focus: Maximize GPU compute utilization

**Megatron-DeepSpeed** = **Best of Both Worlds**
- Combines tensor, pipeline, and data parallelism (3D)
- Can train models >1 trillion parameters
- Used for largest open-source models (BLOOM, GPT-NeoX)
- Industry standard for extreme-scale training

---

## Architecture Overview

### DeepSpeed ZeRO

**Core Philosophy:** Eliminate memory redundancy in data-parallel training

```
Traditional Data Parallel (DDP):
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   GPU 0     │  │   GPU 1     │  │   GPU 2     │
│ Model: 100% │  │ Model: 100% │  │ Model: 100% │
│ Optim: 100% │  │ Optim: 100% │  │ Optim: 100% │
│  Grad: 100% │  │  Grad: 100% │  │  Grad: 100% │
│ Data: Batch0│  │ Data: Batch1│  │ Data: Batch2│
└─────────────┘  └─────────────┘  └─────────────┘
Total Memory: 300% (3x redundancy)

DeepSpeed ZeRO-3:
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   GPU 0     │  │   GPU 1     │  │   GPU 2     │
│ Model: 33%  │  │ Model: 33%  │  │ Model: 33%  │
│ Optim: 33%  │  │ Optim: 33%  │  │ Optim: 33%  │
│  Grad: 33%  │  │  Grad: 33%  │  │  Grad: 33%  │
│ Data: Batch0│  │ Data: Batch1│  │ Data: Batch2│
└─────────────┘  └─────────────┘  └─────────────┘
Total Memory: 100% (no redundancy)
```

**ZeRO Stages:**
- **ZeRO-1:** Partition optimizer states (4x memory reduction)
- **ZeRO-2:** + Partition gradients (8x memory reduction)
- **ZeRO-3:** + Partition parameters (N× reduction, N = #GPUs)
- **ZeRO-Infinity:** + CPU/NVMe offload (theoretically unlimited)

### Megatron-LM Tensor Parallelism

**Core Philosophy:** Split model computation across GPUs for efficiency

```
Tensor Parallelism (TP) - Layer-wise Split:

Standard Single-GPU:
Input → [Linear Layer (full)] → Output

Tensor Parallel (2 GPUs):
           ┌─ GPU 0: [Linear_shard_0] ─┐
Input ────┤                             ├─ Concat → Output
           └─ GPU 1: [Linear_shard_1] ─┘

Example: GPT-3 MLP Layer (hidden_size = 12288)
┌────────────────────────────────────────────┐
│  Standard (1 GPU):                         │
│  Linear: [12288, 49152]  = 600M params     │
└────────────────────────────────────────────┘

┌────────────────────────────────────────────┐
│  Tensor Parallel (4 GPUs):                 │
│  GPU 0: [12288, 12288] = 150M params       │
│  GPU 1: [12288, 12288] = 150M params       │
│  GPU 2: [12288, 12288] = 150M params       │
│  GPU 3: [12288, 12288] = 150M params       │
└────────────────────────────────────────────┘
```

**Communication Pattern:**
```python
# Forward pass (Column-wise parallelism)
# All-gather not needed, each GPU has same input
Y_local = Linear_local(X)  # Independent computation
# All-reduce to combine outputs
Y = All_Reduce(Y_local)

# Backward pass
# Gradient flows back through all-reduce
dX = Linear_local.backward(dY)
```

**Megatron's Transformer Layer Parallelism:**
```
                  Attention
                     |
              ┌──────┴──────┐
         GPU 0: Q,K,V    GPU 1: Q,K,V
         (head 0-15)     (head 16-31)
              |              |
         Attention_0    Attention_1
              |              |
              └──────┬──────┘
                All-Reduce
                     |
                   MLP
              ┌──────┴──────┐
         GPU 0: FC1      GPU 1: FC1
         (half dims)     (half dims)
              |              |
              └──────┬──────┘
                All-Reduce
```

### Pipeline Parallelism (Both Support)

**Splits model vertically (by layers):**

```
4-Stage Pipeline (16 layers total):
┌─────────────────────────────────────────────┐
│ GPU 0: Layers  0-3  (Embedding + L0-L3)     │
│ GPU 1: Layers  4-7                          │
│ GPU 2: Layers  8-11                         │
│ GPU 3: Layers 12-15 (L12-L15 + Head)        │
└─────────────────────────────────────────────┘

Execution (GPipe schedule):
Time →
GPU 0: [F0][F1][F2][F3]          [B0][B1][B2][B3]
GPU 1:     [F0][F1][F2][F3]    [B0][B1][B2][B3]
GPU 2:         [F0][F1][F2][F3][B0][B1][B2][B3]
GPU 3:             [F0][F1][F2][F3][B0][B1][B2][B3]

F = Forward pass, B = Backward pass
Numbers = Micro-batch ID

Pipeline Bubble (idle time): ~25% with 4 stages
```

**DeepSpeed PipeDream:**
- 1F1B schedule (less memory, less bubble)
- Gradient accumulation across pipeline
- Supports heterogeneous stages

**Megatron Pipeline:**
- Interleaved schedules for reduced bubble
- Virtual pipeline stages
- Memory-efficient schedules

---

## Feature Comparison

### Core Parallelism Strategies

| Feature | DeepSpeed | Megatron-LM | Winner |
|---------|-----------|-------------|--------|
| **Data Parallelism** | ✅ ZeRO | ✅ Basic | DeepSpeed |
| **Tensor Parallelism** | ⚠️ Via Megatron | ✅ Core feature | Megatron |
| **Pipeline Parallelism** | ✅ PipeDream | ✅ GPipe + Virtual | Tie |
| **3D Parallelism** | ⚠️ Manual | ⚠️ Manual | Tie (both complex) |
| **Sequence Parallelism** | ❌ No | ✅ Yes | Megatron |
| **Context Parallelism** | ❌ No | ✅ Yes (recent) | Megatron |

### Memory Optimization

| Feature | DeepSpeed | Megatron-LM | Notes |
|---------|-----------|-------------|-------|
| **Parameter Sharding** | ✅ ZeRO-3 | ⚠️ Via TP only | DeepSpeed more flexible |
| **Optimizer Sharding** | ✅ ZeRO-1+ | ⚠️ Via TP only | DeepSpeed automatic |
| **Gradient Sharding** | ✅ ZeRO-2+ | ⚠️ Via TP only | DeepSpeed automatic |
| **CPU Offload** | ✅ Full support | ❌ No | DeepSpeed |
| **NVMe Offload** | ✅ ZeRO-Infinity | ❌ No | DeepSpeed |
| **Activation Checkpointing** | ✅ Yes | ✅ Yes | Tie |
| **Activation Offload** | ✅ CPU offload | ❌ No | DeepSpeed |

### Computation Optimization

| Feature | DeepSpeed | Megatron-LM | Notes |
|---------|-----------|-------------|-------|
| **Fused Kernels** | ✅ Some | ✅ Extensive | Megatron |
| **FlashAttention** | ✅ Supported | ✅ Optimized | Tie |
| **Kernel Fusion** | ⚠️ Limited | ✅ Many fused ops | Megatron |
| **Mixed Precision** | ✅ FP16/BF16 | ✅ FP16/BF16 | Tie |
| **FP8 Training** | ⚠️ Experimental | ✅ Transformer Engine | Megatron |
| **Gradient Compression** | ✅ 1-bit Adam | ❌ No | DeepSpeed |

### Usability

| Feature | DeepSpeed | Megatron-LM | Notes |
|---------|-----------|-------------|-------|
| **Model Flexibility** | ✅ Any PyTorch | ⚠️ GPT/BERT/T5 only | DeepSpeed |
| **Config-Driven** | ✅ JSON config | ⚠️ CLI args | DeepSpeed simpler |
| **Code Changes** | ✅ Minimal | ❌ Significant | DeepSpeed |
| **HuggingFace Integration** | ✅ Excellent | ⚠️ Manual | DeepSpeed |
| **Checkpointing** | ✅ Automatic | ✅ Custom | DeepSpeed easier |
| **Profiling** | ✅ FLOPs profiler | ⚠️ Manual | DeepSpeed |

### Performance Features

| Feature | DeepSpeed | Megatron-LM | Winner |
|---------|-----------|-------------|--------|
| **Communication Overlap** | ✅ Yes | ✅ Yes | Tie |
| **Gradient Accumulation** | ✅ Automatic | ✅ Manual | DeepSpeed |
| **Dynamic Loss Scaling** | ✅ Yes | ✅ Yes | Tie |
| **Distributed Optimizer** | ✅ ZeRO | ⚠️ Basic | DeepSpeed |
| **Custom All-Reduce** | ⚠️ Standard | ✅ Optimized | Megatron |

---

## Megatron-DeepSpeed Integration

### Why Combine Them?

**DeepSpeed strengths:**
- Memory efficiency (ZeRO)
- Easy to use
- Flexible optimizer options
- CPU/NVMe offload

**Megatron strengths:**
- Tensor parallelism (compute efficiency)
- Fused CUDA kernels
- Optimized Transformer architecture
- Sequence parallelism

**Together:**
- Train 100B-1T+ parameter models
- Optimal memory AND compute efficiency
- Industry-proven (BLOOM, GPT-NeoX, Jurrasic-1)

### Architecture

```
Megatron-DeepSpeed: 3D Parallelism
┌──────────────────────────────────────────────────────────┐
│                    Data Parallel (DP)                     │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────┐ │
│  │  DP Group 0    │  │  DP Group 1    │  │  DP Group 2 │ │
│  │ ┌────────────┐ │  │ ┌────────────┐ │  │ ┌──────────┐│ │
│  │ │Pipeline 0  │ │  │ │Pipeline 0  │ │  │ │Pipeline 0││ │
│  │ │┌──┬──┬──┐  │ │  │ │┌──┬──┬──┐  │ │  │ │┌──┬──┬──┐││ │
│  │ ││TP│TP│TP│  │ │  │ ││TP│TP│TP│  │ │  │ ││TP│TP│TP│││ │
│  │ ││0 │1 │2 │  │ │  │ ││0 │1 │2 │  │ │  │ ││0 │1 │2 │││ │
│  │ │└──┴──┴──┘  │ │  │ │└──┴──┴──┘  │ │  │ │└──┴──┴──┘││ │
│  │ │┌──────────┐ │  │ │┌──────────┐ │  │ │┌──────────┐││ │
│  │ ││Pipeline 1│ │  │ ││Pipeline 1│ │  │ ││Pipeline 1│││ │
│  │ │└──────────┘ │  │ │└──────────┘ │  │ │└──────────┘││ │
│  │ └────────────┘ │  │ └────────────┘ │  │ └──────────┘│ │
│  └────────────────┘  └────────────────┘  └─────────────┘ │
└──────────────────────────────────────────────────────────┘

Example: 64 GPUs
- Data Parallel:     4 groups (ZeRO sharding within each)
- Pipeline Parallel: 4 stages
- Tensor Parallel:   4 GPUs
- Total: 4 × 4 × 4 = 64 GPUs
```

### Setup Example

```bash
# Clone Megatron-DeepSpeed
git clone https://github.com/microsoft/Megatron-DeepSpeed.git
cd Megatron-DeepSpeed

# Training GPT-3 175B with 3D parallelism
GPUS_PER_NODE=8
MASTER_ADDR=node0
MASTER_PORT=6000
NNODES=32  # 256 GPUs total
NODE_RANK=0

TP_SIZE=8          # Tensor parallel size
PP_SIZE=16         # Pipeline parallel size
DP_SIZE=2          # Data parallel size (256/(8*16)=2)

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --num-layers 96 \
    --hidden-size 12288 \
    --num-attention-heads 96 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 1 \
    --global-batch-size 1536 \
    --train-iters 500000 \
    --lr 0.00012 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --bf16
"

DEEPSPEED_ARGS="
    --deepspeed \
    --deepspeed_config ds_config.json \
    --zero-stage 1 \
    --deepspeed-activation-checkpointing
"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
    pretrain_gpt.py \
    $GPT_ARGS \
    $DEEPSPEED_ARGS \
    --data-path my-gpt3_00_text_document \
    --vocab-file gpt2-vocab.json \
    --merge-file gpt2-merges.txt \
    --save-interval 1000 \
    --save checkpoints/gpt3-175b \
    --load checkpoints/gpt3-175b \
    --tensorboard-dir tensorboard
```

### DeepSpeed Config for Megatron-DeepSpeed

```json
{
  "train_batch_size": 1536,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": "auto",

  "bf16": {"enabled": true},

  "zero_optimization": {
    "stage": 1,
    "reduce_bucket_size": 500000000,
    "allgather_bucket_size": 500000000,
    "overlap_comm": true,
    "contiguous_gradients": true
  },

  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": true,
    "synchronize_checkpoint_boundary": false
  },

  "gradient_clipping": 1.0,

  "steps_per_print": 10,
  "wall_clock_breakdown": false
}
```

**Key Points:**
- Use ZeRO-1 or ZeRO-2 (not ZeRO-3) with tensor parallelism
- ZeRO-3 conflicts with tensor parallelism (redundant sharding)
- Activation checkpointing is critical for large models
- `train_batch_size` = micro_batch × grad_accum × DP × num_nodes

---

## Performance Benchmarks

### Single-Node Performance (8x A100 80GB)

#### GPT-3 6.7B (GPT-J equivalent)

| Configuration | Throughput (tokens/s) | Memory/GPU | Efficiency |
|---------------|----------------------|------------|------------|
| **DeepSpeed ZeRO-2** | 11,200 | 65GB | 100% |
| **Megatron TP=2** | 13,400 | 68GB | 120% |
| **Megatron TP=4** | 14,800 | 72GB | 132% |
| **Megatron TP=8** | 12,600 | 76GB | 113% |
| **Megatron-DS (TP=4, ZeRO-1)** | 15,200 | 58GB | 136% |

**Insights:**
- Megatron TP=4 sweet spot for 8 GPUs (best compute/comm balance)
- TP=8 over-parallelizes (too much communication)
- Megatron-DS combines best: TP for compute + ZeRO for memory

#### GPT-3 20B

| Configuration | Feasible? | Memory/GPU | Throughput |
|---------------|-----------|------------|------------|
| **DeepSpeed ZeRO-2** | ❌ OOM | OOM | N/A |
| **DeepSpeed ZeRO-3** | ✅ Yes | 76GB | 4,200 tok/s |
| **Megatron TP=8** | ✅ Yes | 78GB | 5,800 tok/s |
| **Megatron TP=4 + ZeRO-2** | ✅ Yes | 62GB | 6,400 tok/s |

**Insights:**
- Pure ZeRO-3 works but slower (too much communication)
- Megatron TP=8 faster but uses more memory
- Combining TP + ZeRO optimal (best speed + memory)

### Multi-Node Performance

#### GPT-3 175B on 64 A100 GPUs (8 nodes)

| Configuration | TP | PP | DP | Throughput | MFU* |
|---------------|----|----|----|-----------|----|
| **DeepSpeed ZeRO-3** | 1 | 1 | 64 | 1,800 tok/s | 28% |
| **Megatron (TP only)** | 64 | 1 | 1 | OOM | N/A |
| **Megatron (TP + PP)** | 8 | 8 | 1 | 3,200 tok/s | 51% |
| **Megatron-DS (3D)** | 8 | 4 | 2 | 4,100 tok/s | 66% |

*MFU = Model FLOPs Utilization (% of theoretical peak)

**Configuration Details:**
```
Megatron-DS (best):
- TP = 8 (intra-node, NVLink bandwidth)
- PP = 4 (inter-node, reduce bubble)
- DP = 2 (ZeRO-1 sharding)
- Total: 8 × 4 × 2 = 64 GPUs
```

**Why this works:**
- Tensor parallelism within node (fast NVLink)
- Pipeline parallelism across nodes (tolerate slower IB)
- Data parallelism for batch scaling + ZeRO memory savings

#### GPT-3 175B on 256 A100 GPUs (32 nodes)

| Configuration | Throughput | Scaling Efficiency | Cost/Token |
|---------------|------------|-------------------|------------|
| **Megatron (TP=8, PP=32)** | 10,200 tok/s | 79% | $0.052 |
| **Megatron-DS (TP=8, PP=16, DP=2)** | 14,800 tok/s | 90% | $0.036 |
| **Megatron-DS + 1-bit Adam** | 16,400 tok/s | 100%* | $0.032 |

*Super-linear due to reduced communication bottleneck

**Key Takeaway:** Megatron-DS with 1-bit Adam is state-of-the-art for extreme scale

### Sequence Length Scaling

#### LLaMA 13B with Different Sequence Lengths

| Seq Length | DeepSpeed | Megatron | Megatron-DS |
|------------|-----------|----------|-------------|
| **512** | 11,200 tok/s | 12,400 tok/s | 13,100 tok/s |
| **1024** | 10,800 tok/s | 12,100 tok/s | 12,900 tok/s |
| **2048** | 9,600 tok/s | 11,800 tok/s | 12,500 tok/s |
| **4096** | 7,200 tok/s | 10,600 tok/s | 11,200 tok/s |
| **8192** | 4,100 tok/s | 8,900 tok/s | 9,600 tok/s |

**Insights:**
- Megatron's sequence parallelism helps at long context
- DeepSpeed degrades faster (all-gather overhead)
- Megatron-DS maintains performance best

---

## Code Examples

### Pure DeepSpeed

```python
import deepspeed
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2-xl")

ds_config = {
    "train_batch_size": 128,
    "train_micro_batch_size_per_gpu": 4,
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"}
    }
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)

# Training loop
for batch in dataloader:
    loss = model_engine(**batch).loss
    model_engine.backward(loss)
    model_engine.step()
```

### Pure Megatron-LM

Megatron requires custom model implementation (not drop-in):

```python
# From Megatron-LM/megatron/model/gpt_model.py
from megatron import get_args, mpu
from megatron.model import GPTModel, GPTModelPipe
from megatron.training import pretrain

def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    args = get_args()

    model = GPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    # Custom dataset loading
    pass

def forward_step(data_iterator, model):
    """Forward training step."""
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, lambda x: x

# Main training
if __name__ == "__main__":
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
    )
```

**Launch:**
```bash
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    pretrain_gpt.py \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 2 \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --micro-batch-size 4 \
    --global-batch-size 128 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --train-iters 500000 \
    --lr 2.5e-4 \
    --clip-grad 1.0 \
    --bf16 \
    --split 969,30,1 \
    --data-path my-gpt3_text_document
```

### Megatron-DeepSpeed

```python
# Similar to pure Megatron, but add DeepSpeed initialization
from megatron.training import pretrain
import deepspeed

def model_provider(pre_process=True, post_process=True):
    model = GPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model

def train_valid_test_datasets_provider(train_val_test_num_samples):
    # Build datasets
    pass

def forward_step(data_iterator, model):
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    return output_tensor, lambda x: x

if __name__ == "__main__":
    # Megatron handles DeepSpeed initialization internally
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
    )
```

**Launch with DeepSpeed:**
```bash
deepspeed --num_gpus=8 \
    --num_nodes=4 \
    --hostfile=hostfile \
    pretrain_gpt.py \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 4 \
    --num-layers 48 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --micro-batch-size 2 \
    --global-batch-size 512 \
    --seq-length 2048 \
    --train-iters 500000 \
    --lr 1.2e-4 \
    --clip-grad 1.0 \
    --bf16 \
    --deepspeed \
    --deepspeed_config ds_config.json \
    --zero-stage 1 \
    --deepspeed-activation-checkpointing \
    --data-path my-gpt3_text_document
```

**ds_config.json:**
```json
{
  "train_batch_size": 512,
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": "auto",

  "bf16": {"enabled": true},

  "zero_optimization": {
    "stage": 1,
    "reduce_bucket_size": 500000000,
    "allgather_bucket_size": 500000000
  },

  "activation_checkpointing": {
    "partition_activations": true,
    "contiguous_memory_optimization": true
  },

  "gradient_clipping": 1.0,
  "wall_clock_breakdown": false
}
```

---

## Use Case Recommendations

### Decision Matrix

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| **Fine-tuning <7B models** | DeepSpeed | Easier, sufficient performance |
| **Fine-tuning 7B-13B** | DeepSpeed ZeRO-3 | Memory efficiency, easy setup |
| **Fine-tuning 13B-70B** | Megatron-DS | TP + ZeRO for best efficiency |
| **Pre-training <20B** | DeepSpeed | Simpler, works well |
| **Pre-training 20B-100B** | Megatron-DS | 3D parallelism necessary |
| **Pre-training >100B** | Megatron-DS | Only viable option |
| **Custom architectures** | DeepSpeed | Megatron limited to GPT/BERT/T5 |
| **GPT/BERT at scale** | Megatron-DS | Optimized kernels, proven |
| **Research experiments** | DeepSpeed | Flexibility, easy iteration |
| **Production (>50B)** | Megatron-DS | Best performance, proven scale |
| **HuggingFace models** | DeepSpeed | Seamless integration |
| **Long sequences (>8k)** | Megatron-DS | Sequence parallelism |
| **Limited GPU memory** | DeepSpeed | CPU/NVMe offload |
| **High GPU compute** | Megatron | Tensor parallelism efficiency |

### Model Size Recommendations

```
<1B params:     DeepSpeed ZeRO-1 or standard DDP
1B-7B params:   DeepSpeed ZeRO-2
7B-13B params:  DeepSpeed ZeRO-3
13B-30B params: Megatron-DS (TP=2-4, ZeRO-2)
30B-100B params: Megatron-DS (TP=4-8, PP=2-4, ZeRO-1)
100B-1T params: Megatron-DS (TP=8, PP=8-16, ZeRO-1)
>1T params:     Megatron-DS (TP=8, PP=16+, ZeRO-1) + expert research
```

### Hardware-Specific Recommendations

**Single Node (8x A100):**
- <20B: DeepSpeed ZeRO-3
- 20B-50B: Megatron TP=4-8 or Megatron-DS (TP=4, ZeRO-2)

**Multi-Node (32-64 GPUs):**
- <50B: DeepSpeed ZeRO-3 (unless using TP)
- 50B-175B: Megatron-DS (TP=8, PP=4-8, DP=1-2)

**Multi-Node (>64 GPUs):**
- Any model: Megatron-DS with 3D parallelism
- Use TP within nodes, PP across nodes, DP for batch scaling

### Industry Use Cases

**DeepSpeed Successes:**
- Microsoft Turing-NLG (17B params)
- Hugging Face models (most <20B models)
- Stability AI fine-tuning
- Research labs (flexibility)

**Megatron-LM Successes:**
- NVIDIA Megatron-Turing NLG (530B)
- GPT-NeoX-20B
- BLOOM (176B, with DeepSpeed)
- Internal NVIDIA models

**Megatron-DeepSpeed Successes:**
- BLOOM (176B) by BigScience
- GPT-NeoX (20B) by EleutherAI
- Jurassic-1 (178B) by AI21 Labs
- Academic supercomputer trainings

---

## Advanced Topics

### ZeRO vs Tensor Parallelism: When to Use Which?

**ZeRO (DeepSpeed):**
- **Pros:** Easy, works with any model, excellent memory efficiency
- **Cons:** Communication overhead at large scale, doesn't improve compute
- **Best for:** Memory-constrained, flexibility, ease of use

**Tensor Parallelism (Megatron):**
- **Pros:** Compute efficiency, lower latency, proven at scale
- **Cons:** Requires model rewrite, limited to specific architectures
- **Best for:** Compute-bound, GPT/BERT/T5, multi-node

**Hybrid (Megatron-DeepSpeed):**
- **Pros:** Best of both worlds
- **Cons:** More complex setup
- **Best for:** Extreme scale (>50B params)

### Communication Patterns

**DeepSpeed ZeRO-3:**
```
Forward pass:
1. All-gather parameters for layer_i
2. Compute forward(layer_i)
3. Discard non-owned parameters
4. Repeat for each layer

Backward pass:
1. All-gather parameters for layer_i
2. Compute gradients
3. Reduce-scatter gradients (aggregate & shard)
4. Discard non-owned parameters
5. Repeat for each layer

Communication volume per layer:
- Forward: 1 all-gather (P parameters)
- Backward: 1 all-gather + 1 reduce-scatter (2P parameters)
- Total: 3P parameters per layer
```

**Megatron Tensor Parallelism:**
```
Forward pass (each layer):
1. All-reduce attention outputs
2. All-reduce MLP outputs
Total: 2 all-reduces per layer

Backward pass (each layer):
1. All-reduce attention gradients
2. All-reduce MLP gradients
Total: 2 all-reduces per layer

Communication volume per layer:
- Forward: 2H × B × S (H=hidden_dim, B=batch, S=seq_len)
- Backward: 2H × B × S
- Total: ~activations, not parameters (much smaller!)
```

**Key Insight:** Tensor parallelism communicates activations (small), ZeRO communicates parameters (large). This is why Megatron is faster for compute-bound models.

### Sequence Parallelism (Megatron Feature)

For long sequences (>4096 tokens):

```python
# Standard: Each GPU has full sequence (memory intensive)
# Sequence Parallel: Shard sequence across TP group

# Example: Sequence length = 8192, TP = 4
# Each GPU processes 8192/4 = 2048 tokens

# In LayerNorm/Dropout (non-tensor-parallel layers):
# Shard along sequence dimension instead of replicating

Benefits:
- Reduces activation memory by TP factor
- Enables training on longer sequences
- Critical for context lengths >8k
```

### Optimizing 3D Parallelism Ratios

**General guidelines:**

```python
# Given N GPUs, choose TP, PP, DP such that:
# N = TP × PP × DP

# Rule 1: TP within nodes (use NVLink)
# Typical: TP = 4 or 8 for 8-GPU nodes

# Rule 2: PP across nodes (tolerate slower interconnect)
# Typical: PP = num_nodes / 2 to reduce pipeline bubble

# Rule 3: DP for batch scaling
# DP = N / (TP × PP)

# Example: 128 GPUs (16 nodes × 8 GPUs)
TP = 8   # Within-node
PP = 4   # Across 4 groups of nodes
DP = 4   # 128 / (8 × 4) = 4
```

**Tuning for specific models:**

```
GPT-3 175B on 256 GPUs:
- Option 1: TP=8, PP=32, DP=1  (minimize memory, max pipeline)
- Option 2: TP=8, PP=16, DP=2  (balance, ZeRO-1 for memory)
- Option 3: TP=4, PP=16, DP=4  (more data parallel, larger batch)

Empirical testing shows Option 2 best (lowest bubble, good batch size)
```

### Memory Breakdown Example

**GPT-3 175B on 256 A100 GPUs with Megatron-DS (TP=8, PP=16, DP=2):**

```
Total parameters: 175B × 2 bytes (BF16) = 350GB

Per GPU (with TP=8, PP=16):
- Model parameters: 350GB / (8 × 16) = 2.73GB
- Optimizer states (Adam): 2.73GB × 4 = 10.92GB  (ZeRO-1: /2 = 5.46GB)
- Gradients: 2.73GB
- Activations (batch=1, seq=2048): ~12GB
- Total: 2.73 + 5.46 + 2.73 + 12 = 22.92GB ✅ Fits in 80GB

Without ZeRO-1:
- Total: 2.73 + 10.92 + 2.73 + 12 = 28.38GB (still fits, but tighter)

DeepSpeed ZeRO-3 only (no TP/PP):
- Parameters: 350GB / 256 = 1.37GB
- Optimizer: 1.37GB × 4 = 5.48GB
- Gradients: 1.37GB
- Activations: ~18GB (larger batch needed for efficiency)
- Total: 1.37 + 5.48 + 1.37 + 18 = 26.22GB
- But: Much more communication (slower)
```

---

## Summary

### Quick Decision Guide

```
┌─────────────────────────────────────────┐
│      Model Size < 13B?                  │
│              │                           │
│         Yes  │  No                       │
│              ↓                           │
│         DeepSpeed                        │
│         ZeRO-2/3                         │
└─────────────────────────────────────────┘
              ↓ No
┌─────────────────────────────────────────┐
│      Using GPT/BERT/T5?                 │
│              │                           │
│         Yes  │  No                       │
│              │                           │
│              ↓                           │
│      Model Size < 70B?                  │
│              │                           │
│         Yes  │  No                       │
│              │                           │
│              ↓  ↓                        │
│     Megatron-DS  Megatron-DS            │
│     (TP+ZeRO)    (3D Parallelism)       │
│                                          │
│              ↓ No (custom model)        │
│         DeepSpeed ZeRO-3                │
│         (only option)                   │
└─────────────────────────────────────────┘
```

### Final Recommendations

**Use DeepSpeed when:**
- Model <13B parameters
- Custom model architecture
- Need CPU/NVMe offload
- Prioritize ease of use
- HuggingFace integration important

**Use Megatron-LM when:**
- Training GPT/BERT/T5 from scratch
- Need maximum compute efficiency
- Have access to NVIDIA GPUs with NVLink
- Can invest in custom model implementation

**Use Megatron-DeepSpeed when:**
- Training models >30B parameters
- Pre-training at extreme scale
- Need 3D parallelism
- Following industry best practices (BLOOM, GPT-NeoX)

### Performance Summary

| Model Size | Best Framework | Typical Setup |
|------------|----------------|---------------|
| <1B | DeepSpeed or DDP | ZeRO-1 or None |
| 1B-7B | DeepSpeed | ZeRO-2 |
| 7B-13B | DeepSpeed | ZeRO-3 |
| 13B-30B | Megatron-DS | TP=2-4, ZeRO-2 |
| 30B-100B | Megatron-DS | TP=4-8, PP=2-4, ZeRO-1 |
| 100B-1T | Megatron-DS | TP=8, PP=8-16, DP=2-4, ZeRO-1 |

---

## Resources

**DeepSpeed:**
- GitHub: https://github.com/microsoft/DeepSpeed
- Docs: https://deepspeed.readthedocs.io/
- Tutorials: https://www.deepspeed.ai/tutorials/

**Megatron-LM:**
- GitHub: https://github.com/NVIDIA/Megatron-LM
- Paper: https://arxiv.org/abs/1909.08053 (original)
- Paper: https://arxiv.org/abs/2104.04473 (Megatron-Turing)

**Megatron-DeepSpeed:**
- GitHub: https://github.com/microsoft/Megatron-DeepSpeed
- BLOOM training: https://huggingface.co/blog/bloom-megatron-deepspeed
- GPT-NeoX: https://github.com/EleutherAI/gpt-neox

---

**Last Updated:** November 2025
