# AutoTP custom patterns example
This example extends the minimal AutoTP script with:

- custom layer sharding patterns (`partition_config`)
- a small text dataset and tokenizer
- a DP-rank random sampler so each DP rank sees different samples

The TP ranks inside the same DP group share the same data order.

## Key code (custom patterns)
The config below targets **Phi-4**, which uses fused QKV and a fused
`gate_up_proj` FFN weight. Both require a `shape` to describe the
sub-parameters inside the fused tensor. Adjust patterns and shapes if your
model uses different names or fused layouts.

```python
ds_config = {
    "zero_optimization": {"stage": 2},
    "tensor_parallel": {
        "autotp_size": args.tp_size,
        "partition_config": {
            "use_default_specs": False,
            "layer_specs": [
                {
                    "patterns": [".*\\.self_attn\\.qkv_proj\\.weight$"],
                    "partition_type": "column",
                    "shape": (3, -1),  # [Q, K, V] fused on dim 0
                    "partition_dim": 0,  # shard along fused dim 0
                },
                {
                    "patterns": [".*\\.self_attn\\.o_proj\\.weight$"],
                    "partition_type": "row",
                },
                {
                    "patterns": [".*\\.mlp\\.gate_up_proj\\.weight$"],
                    "partition_type": "column",
                    "shape": (2, -1),  # [gate, up] fused on dim 0
                    "partition_dim": 0,  # shard along fused dim 0
                },
                {
                    "patterns": [".*\\.mlp\\.down_proj\\.weight$"],
                    "partition_type": "row",
                },
            ],
        },
    },
    "data_parallel_size": args.dp_size,
}
```

The diagram below illustrates how the fused `gate_up_proj` tensor is split
into sub-parameters for sharding:

![Fused gate_up_proj sub-parameters](./autotp-subparams-gate-up.png)

## How to run
Pick a world size where `tp_size * dp_size = world_size`.

```bash
deepspeed --num_gpus 8 autotp_custom_patterns.py \
  --model_name microsoft/Phi-4-multimodal-instruct \
  --tp_size 4 \
  --dp_size 2 \
  --seq_length 512 \
  --num_steps 20
```

`torchrun` also works if you prefer the PyTorch launcher.

