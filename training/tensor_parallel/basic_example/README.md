# AutoTP training (Tensor Parallel)

This directory documents the AutoTP training API for tensor-parallel sharding
during training. AutoTP recognizes typical parameter patterns and
automatically applies proper partitioning.

## Overview

This example provides a compact AutoTP + ZeRO-2 training script,
`autotp_example.py`. It focuses on the AutoTP + ZeRO-2 flow and keeps only the
pieces required to launch AutoTP:

- create TP/DP process groups
- enable AutoTP before model creation
- shard with `deepspeed.tp_model_init`
- initialize DeepSpeed with `tensor_parallel.autotp_size`

The example feeds synthetic token batches (broadcast within each TP group) so
you can validate the AutoTP setup without extra dataset plumbing.

AutoTP recognizes supported model architectures (for example, Llama) and
automatically partitions parameters, so you do not need to specify any manual
partitioning rules for those models. If your model is not supported by AutoTP,
refer to the
[custom layer pattern guide](../custom_patterns/)
for custom layer pattern configuration.

## Key code (AutoTP path)
The core setup mirrors the verification script but is trimmed down:

```python
from deepspeed.module_inject.layers import set_autotp_mode

set_autotp_mode(training=True)
model = AutoModelForCausalLM.from_pretrained(args.model_name)
model = deepspeed.tp_model_init(
    model,
    tp_size=args.tp_size,
    dtype=dtype,
    tp_group=tp_group,
)

ds_config = {
    "train_batch_size": args.batch_size * args.dp_size,
    "train_micro_batch_size_per_gpu": args.batch_size,
    "zero_optimization": {"stage": args.zero_stage},
    "tensor_parallel": {"autotp_size": args.tp_size},
    "data_parallel_size": args.dp_size,
}

mpu = ModelParallelUnit(tp_group, dp_group, args.tp_size, args.dp_size, tp_rank, dp_rank)
engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_config, mpu=mpu)
```

## How to run
Pick a world size where `tp_size * dp_size = world_size`.

```bash
# 8 GPUs: TP=4, DP=2 (AutoTP + ZeRO-2)
deepspeed --num_gpus 8 autotp_example.py \
  --model_name meta-llama/Llama-3.1-8B \
  --tp_size 4 \
  --dp_size 2 \
  --zero_stage 2 \
  --batch_size 1 \
  --seq_length 1024 \
  --num_steps 10
```

`torchrun` works as well if you prefer the PyTorch launcher.

For a smaller test, reduce the world size and TP/DP sizes together:

```bash
deepspeed --num_gpus 2 autotp_example.py \
  --model_name meta-llama/Llama-3.1-8B \
  --tp_size 2 \
  --dp_size 1 \
  --num_steps 5
```

