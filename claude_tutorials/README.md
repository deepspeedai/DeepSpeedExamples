# DeepSpeed ZeRO Implementation and Configuration Mapping

This directory contains comprehensive tutorials and documentation mapping DeepSpeed's ZeRO memory optimization stages directly to code and configuration.

## Directory Structure

```
claude_tutorials/
├── README.md                          # This file
├── annotated_scripts/                  # Line-by-line annotated training scripts
│   ├── 01_hello_deepspeed_annotated.py
│   ├── 02_cifar10_annotated.py
│   ├── 03_superoffload_zero3_annotated.py
│   ├── 04_zenflow_zero2_annotated.py
│   ├── 05_deepspeed_chat_sft_annotated.py
│   ├── 06_domino_megatron_annotated.py
│   ├── 07_tensor_parallel_annotated.py
│   └── 08_bing_bert_annotated.py
├── annotated_configs/                  # Annotated DeepSpeed configuration files
│   ├── zero3_nvme_offload_annotated.json
│   ├── zero3_cpu_offload_annotated.json
│   └── zero2_zenflow_annotated.json
└── guides/                             # Comprehensive reference guides
    ├── ZeRO3_Concept_to_Code.md
    └── Distributed_Training_Guide.md

```

## Overview of Selected Examples

### 1. HelloDeepSpeed (`01_hello_deepspeed_annotated.py`)
**Location:** `training/HelloDeepSpeed/train_bert_ds.py`
**Purpose:** Basic tutorial demonstrating BERT MLM training with DeepSpeed
**Features:**
- Shows all ZeRO stages (0-3) configuration
- Demonstrates `deepspeed.initialize()` API
- Model checkpointing with DeepSpeed
- Integration with PyTorch DataLoader

### 2. CIFAR-10 (`02_cifar10_annotated.py`)
**Location:** `training/cifar/cifar10_deepspeed.py`
**Purpose:** Simple CNN training example
**Features:**
- Configurable ZeRO stages (0-3)
- MoE (Mixture of Experts) support
- Mixed precision training (FP16/BF16/FP32)
- Minimal codebase for understanding basics

### 3. SuperOffload ZeRO-3 (`03_superoffload_zero3_annotated.py`)
**Location:** `training/DeepSpeed-SuperOffload/finetune_zero3.py`
**Purpose:** LLM fine-tuning with ZeRO-3 and SuperOffload
**Features:**
- ZeRO-3 parameter partitioning
- CPU optimizer (DeepSpeedCPUAdam)
- Activation checkpointing/gradient checkpointing
- Flash Attention 2 integration
- Performance metrics (TFLOPS, tokens/sec)

### 4. ZenFlow ZeRO-2 (`04_zenflow_zero2_annotated.py`)
**Location:** `training/DeepSpeed-ZenFlow/finetuning/finetune_llama.py`
**Purpose:** LLaMA fine-tuning with ZeRO-2 and ZenFlow optimizer offloading
**Features:**
- ZeRO-2 optimizer + gradient partitioning
- ZenFlow: Sparse optimizer state updates
- CPU offloading with overlap
- Simple training script

### 5. DeepSpeed-Chat SFT (`05_deepspeed_chat_sft_annotated.py`)
**Location:** `applications/DeepSpeed-Chat/training/step1_supervised_finetuning/main.py`
**Purpose:** RLHF Step 1 - Supervised Fine-Tuning
**Features:**
- Production-ready training pipeline
- LoRA (Low-Rank Adaptation) support
- Dynamic DeepSpeed config generation
- ZeRO-3 model saving utilities

### 6. Domino + Megatron (`06_domino_megatron_annotated.py`)
**Location:** `training/DeepSpeed-Domino/pretrain_gpt.py`
**Purpose:** GPT-3 pre-training with Megatron-LM integration
**Features:**
- Tensor parallelism with DeepSpeed
- Megatron-LM model architecture
- Pipeline parallelism support
- Custom forward step implementation

### 7. Tensor Parallel (`07_tensor_parallel_annotated.py`)
**Location:** `training/tensor_parallel/train.py`
**Purpose:** Tensor parallelism example
**Features:**
- ZeRO-1 with tensor parallelism
- Multi-dimensional parallelism
- Stanford Alpaca fine-tuning

### 8. Bing BERT (`08_bing_bert_annotated.py`)
**Location:** `training/bing_bert/deepspeed_train.py`
**Purpose:** Production-scale BERT pre-training
**Features:**
- Achieved fastest BERT training record (44 min on 1024 V100s)
- Custom dataset provider
- DeepSpeed checkpointing
- Gradient accumulation boundaries

## Configuration Files

### ZeRO-3 with NVMe Offload (`zero3_nvme_offload_annotated.json`)
**Location:** `inference/sglang/ds_offload_nvme_aio.json`
**Key Features:**
- Parameter offloading to NVMe storage
- Async I/O (AIO) configuration
- Auto-tuning parameters
- Buffer management

### ZeRO-3 with CPU Offload (`zero3_cpu_offload_annotated.json`)
**Location:** `inference/sglang/ds_offload_cpu.json`
**Key Features:**
- Parameter offloading to CPU memory
- Pin memory for faster transfers
- Stage 3 optimization settings

### ZeRO-2 with ZenFlow (`zero2_zenflow_annotated.json`)
**Location:** `training/DeepSpeed-ZenFlow/finetuning/zf_config.json`
**Key Features:**
- Optimizer state offloading to CPU
- ZenFlow sparse optimization
- Overlap communication with computation

## Concept Guides

### ZeRO-3 Concept-to-Code Reference
**File:** `guides/ZeRO3_Concept_to_Code.md`

A deep dive into ZeRO Stage 3 optimization:
- Theory of parameter partitioning
- All-Gather operations during forward/backward passes
- Mapping to DeepSpeed source code
- Critical file paths and functions

### Distributed Training Data Flow Guide
**File:** `guides/Distributed_Training_Guide.md`

Complete data flow documentation:
- Single gradient step in ZeRO-3 multi-GPU training
- Parameter sharding and re-assembly
- Gradient reduction across workers
- Parameter update distribution

## How to Use These Materials

### For Learning
1. Start with `01_hello_deepspeed_annotated.py` for basic concepts
2. Progress to `02_cifar10_annotated.py` for a minimal working example
3. Study configuration files to understand ZeRO settings
4. Read the concept guides for theoretical background

### For Implementation
1. Choose the example closest to your use case
2. Review the annotated script to understand key integration points
3. Adapt the configuration file for your model size and hardware
4. Reference the guides for troubleshooting and optimization

### For Debugging
1. Check `Distributed_Training_Guide.md` for data flow understanding
2. Verify configuration against annotated config files
3. Review initialization sequence in annotated scripts
4. Compare your implementation with similar examples

## Key DeepSpeed Concepts Covered

### ZeRO Optimization Stages
- **Stage 0:** Disabled (standard data parallelism)
- **Stage 1:** Optimizer state partitioning
- **Stage 2:** Optimizer + gradient partitioning
- **Stage 3:** Optimizer + gradient + parameter partitioning

### Memory Offloading
- **CPU Offload:** Move optimizer states/parameters to CPU RAM
- **NVMe Offload:** Move parameters to NVMe SSD storage
- **SuperOffload:** Optimized offloading for modern superchips (GH200/GB200/MI300A)

### Communication Optimization
- **Overlap Communication:** Overlap gradient communication with computation
- **Gradient Accumulation:** Accumulate gradients before optimization step
- **All-Gather Buckets:** Batch All-Gather operations for efficiency

### Advanced Features
- **Gradient Checkpointing/Activation Checkpointing:** Trade computation for memory
- **Mixed Precision:** FP16/BF16 training
- **ZenFlow:** Sparse optimizer updates with CPU offloading
- **MoE Support:** Mixture of Experts models

## DeepSpeed Source Code References

The guides map to these critical DeepSpeed source files:

### Core ZeRO Implementation
- `deepspeed/runtime/zero/stage3.py` - ZeRO-3 parameter partitioning
- `deepspeed/runtime/zero/partition_parameters.py` - Parameter sharding logic
- `deepspeed/runtime/zero/partitioned_param_coordinator.py` - All-Gather coordination

### Initialization
- `deepspeed/__init__.py` - Main `initialize()` function
- `deepspeed/runtime/engine.py` - DeepSpeedEngine class

### Offloading
- `deepspeed/runtime/zero/offload_config.py` - Offload configuration
- `deepspeed/ops/adam/cpu_adam.py` - CPU optimizer (DeepSpeedCPUAdam)
- `deepspeed/ops/aio/` - Async I/O for NVMe offloading

## Additional Resources

### Official Documentation
- DeepSpeed Documentation: https://www.deepspeed.ai/
- ZeRO Paper: https://arxiv.org/abs/1910.02054
- ZeRO-Offload Paper: https://arxiv.org/abs/2101.06840
- ZeRO-Infinity Paper: https://arxiv.org/abs/2104.07857

### Example Usage
See the original example directories in the parent repository:
- `training/` - Training examples
- `applications/` - End-to-end applications
- `inference/` - Inference examples
- `benchmarks/` - Performance benchmarks

## Notes

- All annotations are based on the current repository state
- Line numbers reference the original files in the repository
- Configuration values are examples and may need tuning for your hardware
- Some examples require specific datasets or model files

## Contributing

If you find errors or want to suggest improvements, please note them for the repository maintainers.

---

**Created:** 2025-11-18
**Purpose:** Educational materials mapping DeepSpeed ZeRO concepts to implementation
**Target Audience:** ML Engineers, Researchers, DeepSpeed users
