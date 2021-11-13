# Autotuning Hugging Face Examples

This showcases the [autotuning](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/autotuning) feature in DeepSpeed (DS) with Hugging Face (HF) models.

## List of Models

- [DistilBERT](distilbert)
- [BERT-base](bert-base)
- [BERT-large](bert-large)
- [GPT2](gpt2)
- [GPT2-medium](gpt2-medium)
- [GPT2-large](gpt2-large)
- [GPT2-xl](gpt2-xl)
- [DeBERTa](deberta)

Each model folder has a `test_tune.sh` script:

- `./test_tune.sh tune` tunes the model training and then runs it using the selected tuned DeepSpeed configuration.
- `./test_tune.sh 0` runs the model using HF without DeepSpeed.
- `./test_tune.sh z0` runs the model using HF + DS with ZeRO optimization disabled.
- `./test_tune.sh z1` runs the model using HF + DS with ZeRO optimization stage 1.
- `./test_tune.sh z2` runs the model using HF + DS with ZeRO optimization stage 2.
- `./test_tune.sh z3` runs the model using HF + DS with ZeRO optimization stage 3.


## Testing Environment

The training runs on 1 node with 16 Nvidia V100 GPUs. The autotuning uses the same hardware resource as the training.
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

| Model   name | num_params |     baseline (vanila HF)      |          HF + DS hand-tuned          | HF + DS autotuning (fast-mode) | throughput improvement over baseline | autotuning time (mins) | number of experiments |
| :----------: | :--------: | :---------------------------: | :----------------------------------: | :----------------------------: | :----------------------------------: | :--------------------: | :-------------------: |
|  DistilBERT  |    66M     | 5161.902 (gas = 1, mbs = 256) | 5305.067 (z = 0, gas = 1 mbs = 256)  |  5305.067 (z0_gas1_tmbspg256)  |                1.03x                 |           11           |          11           |
|  BERT-base   |   0.11B    | 2502.236 (gas = 1,mbs = 128)  | 2523.684 (z = 0, gas = 1, mbs = 128) |  2736.561 (z0_gas1_tmbspg235)  |                1.09x                 |           35           |          34           |
|  BERT-large  |   0.34B    |  742.692 (gas = 1,mbs = 64)   |  766.929 (z = 1, gas = 1, mbs = 64)  |   808.168 (z1_gas1_tmbspg93)   |                1.09x                 |           36           |          22           |
|     GPT2     |   0.12B    |   284.142 (gas = 1,mbs = 8)   |  397.827 (z = 1, gas = 1, mbs = 8)   |   431.586 (z1_gas1_tmbspg14)   |                1.52x                 |           25           |          17           |
| GPT2-medium  |   0.35B    |   71.61 (gas = 1, mbs = 2)    |  142.211 (z = 1, gas = 1, mbs = 4)   |    163.3 (z1_gas1_tmbspg6)     |                 2.28                 |           15           |          25           |
|  GPT2-large  |   0.77B    |   27.874 (gas = 1, mbs = 1)   |   56.797 (z = 1, gas = 1, mbs = 2)   |    69.061 (z = 1, mbs = 3)     |                2.48x                 |           27           |          13           |
|   GPT2-xl    |    1.5B    |         Not runnable          |      27.462 (gas = 1, mbs = 1)       |    27.497 (z1_gas1_tmbspg1)    |                 inf                  |           21           |           9           |
|   DeBERTa    |    1.5B    |         Not runnable          |   140.587 (z = 1, gas = 1 mbs = 8)   |  162.395  (z1_gas1_tmbspg11)   |                 inf                  |           40           |          12           |
