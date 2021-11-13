# [gpt2](https://huggingface.co/gpt2)

This model has the following configuration:

- 12-layer
- 768 hidden dimension
- 12 attention heads
- 117M parameters.

Refer to [GPT-2/GPT and causal language modeling](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling)

## Environment

The training use fp16 and runs on 1 node with 16 Nvidia V100 GPUs. The autotuning uses the same hardware resource as the training. `max_train_batch_size` is not defined.
The HF packages below are used.

HF examples require installing the `transformers` package from source:
```bash
    git clone https://github.com/huggingface/transformers.git
    cd transformers
    pip install .
```
The `datasets` package can be installed by `pip install datasets`

Below are the versions used in this test.

- transformers (4.12.0)
- datasets (1.11.0)
## Throughput Comparison

The table below shows the throughput (samples per second) comparison. The corresponding train micro-batch size per GPU (mbs or tmbspg) and ZeRO stage used to achieve the throughput value is also shown in the parentheses. Assume the strategy users would use in the handtuning process is to start from `mbs = 1` and increase mbs by 2 each time until running out of GPU memory.
 - `baseline` is the vanila HF without DeepSpeed (DS) and mbs is hand-tuned.
 - `HF + DS hand-tuned` is HF with DS, and mbs is hand-tuned while other DS configuration uses default values.
 - `HF + DS autotuning` is HF with DS, and the DS configuration is selected from autotuning.

Notation: Hugging Face (HF), DeepSpeed (DS), ZeRO stage (z), gradient accumulation steps (gas), train micro-batch size per GPU (mbs or tmbspg).

| Model name | baseline (vanila HF) | HF + DS hand-tuned       | HF + DS autotuning (fast-mode) |
| ---------- | -------------------- | ------------------------ | ------------------------------ |
| GPT2       | 284.142 (mbs = 8)    | 397.827 (z = 1, mbs = 8) | 431.586 (z1_gas1_tmbspg15)     |


## Detailed `HF + DS autotuning` Result Summary

Note that the performance metric used in autotuning is calculated using the timings captured within DeepSpeed forward, backward, and step functions. The sum of these timings is less than the actual training step latency, thus the throughput metric values used by autotuning would be higher than the end-to-end throughput in training.

- Fast-mode Autotuning time: 25 mins
- Number of experiments: 17
- Throughput Improvement over baseline: 1.52x

| tuning_space | num_experiments | best_metric_val | best_exp_name    |
| :----------- | --------------: | --------------: | :--------------- |
| z0           |               9 |         441.693 | z0_gas1_tmbspg11 |
| z1           |               6 |         452.004 | z1_gas1_tmbspg15 |
| z2           |               1 |               0 | z2_gas1_tmbspg15 |
| z3           |               1 |               0 | z3_gas1_tmbspg15 |
| global       |              17 |         452.004 | z1_gas1_tmbspg15 |

Tuning completed in 0:24:19.976427. Total number of experiments: 17.
