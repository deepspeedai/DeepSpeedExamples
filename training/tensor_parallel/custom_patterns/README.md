# AutoTP (Tensor Parallel) Custom Patterns Example

This example extends the minimal AutoTP script with:

- custom layer sharding patterns (`partition_config`)
- a small text dataset and tokenizer
- a DP-rank random sampler so each DP rank sees different samples

The TP ranks inside the same DP group share the same data order.
AutoTP is enabled by the DeepSpeed config (`tensor_parallel.autotp_size`), so
you do not need to call any initialization helpers before `deepspeed.initialize`.

## Key code (custom patterns)

The config below targets **Pythia 6.9B (GPT-NeoX)**, which uses a fused
`query_key_value` projection. We provide a `shape` so AutoTP can split the
fused Q/K/V tensor cleanly across tensor-parallel ranks. The MLP uses
`dense_h_to_4h` / `dense_4h_to_h`, so no extra shape is needed there.

```python
ds_config = {
    "zero_optimization": {"stage": 2},
    "tensor_parallel": {
        "autotp_size": args.tp_size,
        "partition_config": {
            "use_default_specs": False,
            "layer_specs": [
                {
                    "patterns": [".*(self_attention|attention)\\.query_key_value\\.weight$"],
                    "partition_type": "column",
                    "shape": ((q_size, kv_size, kv_size), -1),
                    "partition_dim": 0,
                },
                {
                    "patterns": [".*(self_attention|attention)\\.dense\\.weight$"],
                    "partition_type": "row",
                },
                {
                    "patterns": [".*mlp\\.dense_h_to_4h\\.weight$"],
                    "partition_type": "column",
                },
                {
                    "patterns": [".*mlp\\.dense_4h_to_h\\.weight$"],
                    "partition_type": "row",
                },
            ],
        },
    },
    "data_parallel_size": args.dp_size,
}
```

## How to run
Pick a world size where `tp_size * dp_size = world_size`.

```bash
deepspeed --num_gpus 8 autotp_custom_patterns.py \
  --model_name EleutherAI/pythia-6.9b \
  --tp_size 4 \
  --dp_size 2 \
  --seq_length 512 \
  --num_steps 20
```

`torchrun` also works if you prefer the PyTorch launcher.

