# Compression examples

The legacy BERT, GPT-2, and CIFAR examples were removed because they depended
on `deepspeed.compression`, which was removed in
[deepspeedai/DeepSpeed#8490](https://github.com/deepspeedai/DeepSpeed/pull/8490).

| Example | Description |
| --- | --- |
| [reasoning_aware_compression](reasoning_aware_compression) | One-shot pruning of reasoning LLMs (DeepSeek-R1 distills, Qwen3) calibrated on their own chain-of-thought traces — [RAC, ICLR 2026](https://arxiv.org/abs/2509.12464) |
