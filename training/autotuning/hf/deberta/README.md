# [deberta-v2-xxlarge-mnli](https://huggingface.co/microsoft/deberta-v2-xxlarge)

This model has the following configuration:

- 48-layer
- 1536 hidden dimension
- 1.5B parameters.

Refer to [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://github.com/microsoft/DeBERTa).
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

Notation: Hugging Face (HF), DeepSpeed (DS), ZeRO stage (z), gradient accumulation steps (gas), train micro-batch size per GPU (mbs or tmbspg), reduce_bucket_size (rbs), allgather_bucket_size (abs).

| Model name | baseline (vanila HF) | HF + DS hand-tuned                | HF + DS autotuning (fast-mode) |
| ---------- | -------------------- | --------------------------------- | ------------------------------ |
| DeBERTa    | Not runnable         | 140.587 (z = 1, gas = 1 mbs = 8), | 162.395  (z1_gas1_tmbspg11)    |

## Detailed `HF + DS autotuning` Result Summary

Note that the performance metric used in autotuning is calculated using the timings captured within DeepSpeed forward, backward, and step functions. The sum of these timings is less than the actual training step latency, thus the throughput metric values used by autotuning would be higher than the end-to-end throughput in training.
### Fast-mode
- Autotuning time: 40 mins
- Number of experiments: 12
- Throughput Improvement over baseline: Inf

| tuning_space | num_experiments | best_metric_val | best_exp_name    |
| :----------- | --------------: | --------------: | :--------------- |
| z0           |               1 |               0 | z0_gas1_tmbspg1  |
| z1           |               6 |         177.843 | z1_gas1_tmbspg11 |
| z2           |               4 |         154.002 | z2_gas1_tmbspg14 |
| z3           |               1 |               0 | z3_gas1_tmbspg14 |
| global       |              12 |         177.843 | z1_gas1_tmbspg11 |

Tuning completed in 0:39:25.253998. Total number of experiments: 12.

### Full-mode ("fast" set to false)
- Autotuning time: 1 hr 2 mins
- Number of experiments: 24
- Throughput Improvement over baseline: Inf

| tuning_space      | num_experiments | best_metric_val | best_exp_name                          |
| :---------------- | --------------: | --------------: | :------------------------------------- |
| z0                |               1 |               0 | z0_gas1_tmbspg1                        |
| z1                |               6 |         177.843 | z1_gas1_tmbspg11                       |
| z1_rbs_abs_tmbspg |              12 |         193.577 | z1_rbs5.0e+07_abs1.0e+09_gas1_tmbspg11 |
| z2                |               4 |         154.002 | z2_gas1_tmbspg14                       |
| z3                |               1 |               0 | z3_gas1_tmbspg14                       |
| global            |              24 |         193.577 | z1_rbs5.0e+07_abs1.0e+09_gas1_tmbspg11 |

Tuning completed in 1:02:32.759424. Total number of experiments: 24.
