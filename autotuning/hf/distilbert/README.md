# [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)

This model has the following configuration:

- 12-layer
- 768 hidden dimension
- 12 attention heads
- 66M parameters.

## Environment

The training uses 1 node with 16 Nvidia V100 GPUs, fp32, max_train_batch_size = 4096. The autotuning uses the same hardware resource as the training. `"max_train_batch_size"` is set to `4096`.
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

| Model name | baseline (vanila HF)          | HF + DS hand-tuned                   | HF + DS autotuning (fast-mode) |
| ---------- | ----------------------------- | ------------------------------------ | ------------------------------ |
| DistilBERT | 5161.902 (gas = 1, mbs = 256) | 5305.067 (z = 0, gas = 1 mbs = 256), | 5305.067 (z0_gas1_tmbspg256)   |

3700.296

## Detailed `HF + DS autotuning` Result Summary

Note that the performance metric used in autotuning is calculated using the timings captured within DeepSpeed forward, backward, and step functions. The sum of these timings is less than the actual training step latency, thus the throughput metric values used by autotuning would be higher than the end-to-end throughput in training.

- Fast-mode Autotuning time: 11 mins
- Number of experiments: 11
- Throughput Improvement: 1.03x

| tuning_space | num_experiments | best_metric_val | best_exp_name     |
| :----------- | --------------: | --------------: | :---------------- |
| z0           |               5 |         5759.96 | z0_gas1_tmbspg256 |
| z1           |               2 |         5667.06 | z1_gas1_tmbspg256 |
| z2           |               2 |         5366.97 | z2_gas1_tmbspg256 |
| z3           |               2 |         4892.49 | z3_gas1_tmbspg256 |
| global       |              11 |         5759.96 | z0_gas1_tmbspg256 |

Tuning completed in 0:10:45.085016. Total number of experiments: 11.


| tuning_space | num_experiments | best_metric_val | best_exp_name      |
| :----------- | --------------: | --------------: | :----------------- |
| z0           |               7 |         5759.98 | z0_gas22_tmbspg179 |
| z1           |               2 |         5543.49 | z1_gas1_tmbspg269  |
| z2           |               2 |         5044.88 | z2_gas15_tmbspg269 |
| z3           |               2 |         4627.63 | z3_gas1_tmbspg269  |
| global       |              13 |         5759.98 | z0_gas22_tmbspg179 |

Tuning completed in 0:25:44.502148. Total number of experiments: 13.
