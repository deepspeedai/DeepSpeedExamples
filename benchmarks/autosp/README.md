# AutoSP Benchmarking Examples

This directory contains AutoSP benchmarking examples that demonstrate model compilation and optimization techniques using DeepSpeed and HuggingFace Accelerate. The example script show four compilation modes (AutoSP and baselines) for training large language models:

| Mode | Parallelism Strategy | Execution Backend |
|------|----------------------|-------------------|
| **eager** | Ulysses DistributedAttention | PyTorch Eager |
| **compile** | Ulysses DistributedAttention | PyTorch Inductor |
| **autosp** | Automatic Sequence Parallelism | AutoSP Compiler |
| **ringattn** | RingAttention-style Sequence Parallelism | PyTorch Inductor |

## Files in this Directory

- **run.py**: Benchmarking script with an option to choose either of the 4 compilation modes listed above
- **run_autosp.sh**: Launcher script that configures training runs across multiple GPUs using Hugging Face Accelerate
- **sp_dp_registry.py**: Sequence Parallel and Data Parallel mesh management utilities
- **distributed_attention.py**: Ulysses-styled sequence paralllelism which can be plugged in as an attention backend for HuggingFace
- **ring_attention.py**: Ring Attention algorithm implementation which can be plugged in as an attention backend for HuggingFace
- **configs/**: Training configuration templates for different model sizes and scenarios
- **correctness/**: Correctness validation suite for AutoSP
  - **correctness_run.py**: Runs training for a specific configuration (compile mode, sequence parallel size, ZeRO stage) and saves per-rank losses to a JSON file for comparison
  - **correctness.sh**: Launcher script that orchestrates correctness testing across multiple configurations, running both baseline (compiled Ulysses) and AutoSP modes
  - **validator.py**: Compares per-rank losses between AutoSP and baseline to verify numerical correctness within a configurable threshold

## Setup Guide

Quick start guide to set up the AutoSP example. This example demonstrates usage of AutoSP with [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/index) for distributed training across multiple GPUs.

### Install dependencies

```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

```bash
pip install \
  transformers==4.50.3 \
  tokenizers \
  huggingface-hub \
  safetensors \
  datasets \
  accelerate \
  scipy \
  tqdm \
  pyyaml \
  flash-attn 
```

## Benchmarking

The `benchmarks/autosp/` directory contains for benchmarking scripts:

```bash
cd benchmarks/autosp
```

#### Run autosp on 2 GPUs
```bash
./run_autosp.sh --compile autosp --batch-size 1 --seq-length 64 --sp-size 2 --num-layers 1 --steps 1 --deterministic
```

#### Run eager mode ulysses on 2 GPUs
```bash
./run_autosp.sh --compile eager --batch-size 1 --seq-length 64 --sp-size 2 --num-layers 1 --steps 1 --deterministic
```

#### Run torch.compile'd ulysses on 2 GPUs
```bash
./run_autosp.sh --compile compile --batch-size 1 --seq-length 64 --sp-size 2 --num-layers 1 --steps 1 --deterministic
```

#### Run torch.compile'd ring attention on 2 GPUs
```bash
./run_autosp.sh --compile ringattn --batch-size 1 --seq-length 64 --sp-size 2 --num-layers 1 --steps 1 --deterministic
```

## Correctness Testing

To validate that AutoSP produces numerically correct results matching the baseline, use the correctness test suite:

```bash
cd correctness
./correctness.sh              # Test default sp-sizes: 1, 2, 4, 8
./correctness.sh 2,1          # Test with custom (sp-sizes, dp_size)
```

This runs training for each configuration with both baseline (compiled Ulysses) and AutoSP modes, then compares per-rank losses to verify correctness.

### Expected Output

When running the correctness suite with sp_size=2, you should see output similar to:

```
================================================================
  AutoSP Correctness Test Suite
================================================================
  Configs (sp,dp): 2,1 4,1 8,1
  Zero stages:     0 1
  Steps:           5
  Output dir:      /u/ndani/DeepSpeedExamples/benchmarks/autosp/correctness/output
================================================================

----------------------------------------------------------------
  Test: sp_size=2, dp_size=1, zero_stage=0
----------------------------------------------------------------
  [1/3] Running baseline (--compile compile) ...
  Losses saved: 2 rank(s), 6 step(s) -> /u/ndani/DeepSpeedExamples/benchmarks/autosp/correctness/output/sp2_dp1_zero0/baseline.json
  [2/3] Running autosp  (--compile autosp)  ...
  Losses saved: 2 rank(s), 6 step(s) -> /u/ndani/DeepSpeedExamples/benchmarks/autosp/correctness/output/sp2_dp1_zero0/autosp.json
  [3/3] Validating per-rank losses ...
  PASS (max diff: 3.861427e-03, threshold: 1.000000e-02)

----------------------------------------------------------------
  Test: sp_size=2, dp_size=1, zero_stage=1
----------------------------------------------------------------
  [1/3] Running baseline (--compile compile) ...
  Losses saved: 2 rank(s), 6 step(s) -> /u/ndani/DeepSpeedExamples/benchmarks/autosp/correctness/output/sp2_dp1_zero1/baseline.json
  [2/3] Running autosp  (--compile autosp)  ...
  Losses saved: 2 rank(s), 6 step(s) -> /u/ndani/DeepSpeedExamples/benchmarks/autosp/correctness/output/sp2_dp1_zero1/autosp.json
  [3/3] Validating per-rank losses ...
  PASS (max diff: 3.166199e-03, threshold: 1.000000e-02)

================================================================
  SUMMARY
================================================================
  sp2_dp1_zero0: PASS
  sp2_dp1_zero1: PASS
```

All tests should PASS with loss differences within the configurable threshold (default: 1.0e-2).
