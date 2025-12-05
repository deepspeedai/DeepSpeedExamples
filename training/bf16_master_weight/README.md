# BF16 Low-Precision Master Weights and Optimizer States

This example demonstrates DeepSpeed's new low-precision training options that can significantly reduce memory usage:

- `bf16_master_weights_and_grads`: Keep master parameters and gradients in BF16 instead of FP32
- `bf16_optimizer_states`: Keep optimizer states (e.g., Adam moments) in BF16

These options work with ZeRO Stage 3 and `torch.autocast` to provide memory-efficient training while maintaining numerical stability.

## Memory Savings

Using a 254M parameter simple transformer model with the following configuration:
- Hidden dimension: 1024
- Layers: 12
- Attention heads: 16
- Batch size: 4
- Sequence length: 512
- ZeRO Stage: 3

### 1-GPU Results

| Configuration | Allocated Memory | Peak Memory | Avg Step Time |
|---------------|------------------|-------------|---------------|
| Baseline (fp32 master) | 4.14 GB | 5.93 GB | 0.1042s |
| BF16 low-precision (master + opt states) | **2.71 GB** | 5.73 GB | 0.1121s |

**Allocated memory reduction: 1.43 GB (34.5%)**

### 4-GPU Results (per GPU) - 254M Model

| Configuration | Allocated Memory | Peak Memory | Avg Step Time |
|---------------|------------------|-------------|---------------|
| Baseline (fp32 master) | 1.29 GB | 3.57 GB | 0.1189s |
| BF16 low-precision (master + opt states) | **0.94 GB** | 4.44 GB | 0.1249s |

**Allocated memory reduction: 0.35 GB per GPU (27%)**

### 4-GPU Results (per GPU) - 6.86B Model

Using a 6.86B parameter model (hidden=4096, layers=32, heads=32, batch=1, seq=512):

| Configuration | Allocated Memory | Peak Memory | Avg Step Time |
|---------------|------------------|-------------|---------------|
| Baseline (fp32 master) | 25.74 GB | 41.28 GB | 0.5078s |
| BF16 low-precision (master + opt states) | **16.17 GB** | **33.20 GB** | 0.5064s |

**Memory reduction: 9.57 GB allocated (37%), 8.08 GB peak (19.6%)**

### 4-GPU Results (per GPU) - 6.86B Model with Activation Checkpointing

With activation checkpointing enabled, the optimizer state memory becomes the dominant factor, making the savings even more visible:

| Configuration | Allocated Memory | Peak Memory | Avg Step Time |
|---------------|------------------|-------------|---------------|
| Baseline (fp32 master) | 25.74 GB | 31.38 GB | 0.6016s |
| BF16 low-precision (master + opt states) | **16.17 GB** | **18.93 GB** | 0.6427s |

**Memory reduction: 9.57 GB allocated (37%), 12.45 GB peak (39.7%)**

With activation checkpointing, peak memory drops significantly for both configurations, but the bf16 low-precision option shows an even larger relative improvement - nearly **40% reduction in peak memory**.

The allocated memory reflects the optimizer state memory, which is where the low-precision options provide savings. Peak memory includes activations and temporary buffers which can vary based on execution order.

## Loss Curve Comparison

To verify that BF16 low-precision training maintains numerical stability, we trained for 1000 steps on the Wikitext-103 dataset:

![Loss Comparison](logs/7b_loss_run/loss_comparison.png)

| Configuration | Final Loss | Mean Loss | Loss Std |
|---------------|------------|-----------|----------|
| Baseline (fp32 master) | 3.09 | 2.78 | 1.56 |
| BF16 Low-Precision | 3.12 | 2.90 | 2.37 |

The loss curves show that both configurations converge similarly, demonstrating that the reduced precision does not significantly impact training quality while providing substantial memory savings.

To reproduce the loss curve comparison:

```bash
# Run 1000 steps with wikitext dataset
deepspeed --num_gpus=4 train.py --deepspeed_config configs/baseline.json \
  --num_layers 32 --hidden_dim 4096 --num_heads 32 --batch_size 1 \
  --num_steps 1000 --activation_checkpointing \
  --loss_log_file logs/baseline_loss.csv --use_real_data --seed 42

deepspeed --num_gpus=4 train.py --deepspeed_config configs/bf16_full.json \
  --num_layers 32 --hidden_dim 4096 --num_heads 32 --batch_size 1 \
  --num_steps 1000 --activation_checkpointing \
  --loss_log_file logs/bf16_full_loss.csv --use_real_data --seed 42

# Generate comparison plot
python plot_loss.py --baseline logs/baseline_loss.csv --bf16 logs/bf16_full_loss.csv \
  --output loss_comparison.png
```

## Configuration

### Baseline (FP32 master weights and optimizer states)

```json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3
    }
}
```

### BF16 Low-Precision (BF16 master weights, gradients, and optimizer states)

```json
{
    "bf16": {
        "enabled": true,
        "bf16_master_weights_and_grads": true,
        "bf16_optimizer_states": true
    },
    "zero_optimization": {
        "stage": 3
    },
    "torch_autocast": {
        "enabled": true,
        "dtype": "torch.bfloat16"
    }
}
```

## Usage

### Run Individual Configurations

```bash
# Run baseline configuration
deepspeed --num_gpus=1 train.py --deepspeed_config configs/baseline.json

# Run BF16 low-precision configuration
deepspeed --num_gpus=1 train.py --deepspeed_config configs/bf16_full.json
```

### Run Memory Comparison

```bash
# Run both configurations and generate comparison report
./run_comparison.sh

# With custom settings
./run_comparison.sh --num_layers 24 --hidden_dim 2048 --batch_size 2
```

### Gather Results from Logs

```bash
python gather_memory.py --log_dir logs/<timestamp>
```

## Training Script Options

```
--hidden_dim       Hidden dimension size (default: 1024)
--num_layers       Number of transformer layers (default: 12)
--num_heads        Number of attention heads (default: 16)
--vocab_size       Vocabulary size (default: 50000)
--batch_size       Batch size per GPU (default: 4)
--seq_length       Sequence length (default: 512)
--num_steps        Number of training steps (default: 20)
--warmup_steps     Warmup steps before measuring (default: 5)
--deepspeed_config Path to DeepSpeed config file
```

## Requirements

- DeepSpeed with BF16 support
- PyTorch with BF16 support
- GPU with BF16 support (e.g., NVIDIA Ampere or newer)

## How It Works

### Standard BF16 Training (Baseline)

In standard BF16 training with DeepSpeed:
- Model parameters are stored in BF16
- Forward/backward computations use BF16 via `torch.autocast`
- Master weights are maintained in FP32 for optimizer updates
- Optimizer states (Adam momentum and variance) are in FP32

This requires significant memory for the FP32 copies.

### BF16 Low-Precision Training

With the new options enabled:
- `bf16_master_weights_and_grads=true`: Master weights and gradients stay in BF16
- `bf16_optimizer_states=true`: Adam momentum and variance buffers use BF16

This eliminates the FP32 copies, reducing memory by approximately 2 bytes per parameter for master weights and 4 bytes per parameter for optimizer states (for Adam which has 2 state buffers).

### Memory Breakdown

For a model with N parameters:

| Component | Baseline | BF16 Low-Precision |
|-----------|----------|-------------------|
| Model params | 2N bytes (BF16) | 2N bytes (BF16) |
| Master weights | 4N bytes (FP32) | 2N bytes (BF16) |
| Gradients | 4N bytes (FP32) | 2N bytes (BF16) |
| Adam momentum | 4N bytes (FP32) | 2N bytes (BF16) |
| Adam variance | 4N bytes (FP32) | 2N bytes (BF16) |
| **Total** | **18N bytes** | **10N bytes** |

This gives a theoretical ~44% reduction in optimizer-related memory. The actual savings depend on activation memory and other factors.

## Related Resources

- [DeepSpeed BF16 Documentation](https://www.deepspeed.ai/docs/config-json/#bf16-training-options)
- [DeepSpeed Core API Updates Blog](../../blogs/core_api_update/README.md)
- [Low-precision master params PR](https://github.com/deepspeedai/DeepSpeed/pull/7700)
