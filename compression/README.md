# DeepSpeed Model Compression examples

Examples in this folder are helpful to try out some features and models that take advantage of the DeepSpeed compression library.

A detailed tutorial for understanding and using DeepSpeed model compression features can be seen from here: https://www.deepspeed.ai/tutorials/model-compression/

| Example | Description |
| --- | --- |
| [bert](bert) | Quantization, pruning and layer reduction on BERT (ZeroQuant, XTC) |
| [gpt2](gpt2) | ZeroQuant post-training quantization on GPT-2 |
| [cifar](cifar) | Channel pruning and quantization on a CIFAR ResNet |
| [reasoning_aware_compression](reasoning_aware_compression) | One-shot pruning of reasoning LLMs (DeepSeek-R1 distills, Qwen3) calibrated on their own chain-of-thought traces — [RAC, ICLR 2026](https://arxiv.org/abs/2509.12464) |
