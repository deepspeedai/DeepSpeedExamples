# AutoTP training examples
This folder groups AutoTP training examples at different complexity levels.

## Contents
- `basic_example/`: minimal AutoTP + ZeRO-2 example with synthetic tokens. It also shows that AutoTP recognizes typical parameter patterns and automatically applies proper partitioning.
- `hf_integration/`: Hugging Face Trainer example (adapted from Stanford Alpaca).
- `custom_patterns/`: AutoTP example with custom layer patterns and a simple
  text dataset that uses a DP-rank random sampler. It shows how to define
  parameter partitioning easily for custom models with non-standard parameter
  definitions.

## Related references
- AutoTP training docs: https://github.com/deepspeedai/DeepSpeed/blob/master/docs/code-docs/source/training.rst
- AutoTP training tutorial: https://github.com/deepspeedai/DeepSpeed/blob/master/docs/_tutorials/autotp-training.md
