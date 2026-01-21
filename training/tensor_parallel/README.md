# AutoTP Training Examples

This folder groups AutoTP training examples at different complexity levels.

## Contents
- [Basic example](basic_example): minimal AutoTP + ZeRO-2 example with synthetic tokens. It also shows that AutoTP recognizes typical parameter patterns and automatically applies proper partitioning.
- [HuggingFace integration](hf_integration): Hugging Face Trainer example (adapted from Stanford Alpaca).
- [Custom partitioning patterns](custom_patterns): AutoTP example with custom layer patterns and a simple
  text dataset that uses a DP-rank random sampler. It shows how to define
  parameter partitioning easily for custom models with non-standard parameter
  definitions.

## Related references
- [AutoTP training docs](https://deepspeed.readthedocs.io/en/latest/training.html)
- [AutoTP training tutorial](https://github.com/deepspeedai/DeepSpeed/blob/master/docs/_tutorials/autotp-training.md)
