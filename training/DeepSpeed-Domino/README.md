# Running Tensor Parallel Training with Domino

This example demonstrates how to use Domino for tensor parallel training with large language models such as GPT-3. The setup has been validated on:

 - NVIDIA H200 GPUs using the Docker image: `nvcr.io/nvidia/pytorch:24.12-py3`

 - AMD MI300 GPUs using the Docker image: `rocm/pytorch:rocm6.3.4_ubuntu22.04_py3.10_pytorch_release_2.4.0`

You can pull the same docker images using the following commands:

```
docker pull nvcr.io/nvidia/pytorch:24.12-py3 

docker pull rocm/pytorch:rocm6.3.4_ubuntu22.04_py3.10_pytorch_release_2.4.0
```

## Install Dependencies
```
pip install -r requirements.txt
```

## Prepare the Dataset
Follow the instructions from [Megatron-DeepSpeed](https://github.com/deepspeedai/Megatron-DeepSpeed/tree/main/examples_deepspeed/universal_checkpointing#download-and-pre-process-training-dataset) to prepare the training dataset.

## Launch Training with Domino

Adjust the following parameters in the script as needed:

- **GPUS_PER_NODE**: Number of GPUs per node.
- **VOCAB_FILE**, **MERGE_FILE**, **DATA_PATH**: Paths to the dataset files.
- **--micro-batch-size**: Batch size per GPU.

### Supported Models and Scripts

| Model      | Script                   |
|------------|--------------------------|
| GPT-3 6.7B | `pretrain_gpt3_6.7b.sh`  |
| GPT-3 13B | `pretrain_gpt3_13b.sh`  |



### Example

To train the GPT-3 13B model, run the following command:

```bash
bash pretrain_gpt3_13b.sh
```

Sample output during training:

```
...
iteration: 30 | loss: 10.120 | iteration time (ms): 528.60 
iteration: 31 | loss: 9.984 | iteration time (ms): 527.02 
iteration: 32 | loss: 9.751 | iteration time (ms): 521.55 
iteration: 33 | loss: 9.496 | iteration time (ms): 525.22 
iteration: 34 | loss: 9.510 | iteration time (ms): 523.22 
iteration: 35 | loss: 9.551 | iteration time (ms): 527.20 
iteration: 36 | loss: 9.549 | iteration time (ms): 525.23 
iteration: 37 | loss: 9.204 | iteration time (ms): 527.17 
iteration: 38 | loss: 9.215 | iteration time (ms): 524.86 
iteration: 39 | loss: 9.091 | iteration time (ms): 525.64 
iteration: 40 | loss: 8.950 | iteration time (ms): 523.91 
iteration: 41 | loss: 8.773 | iteration time (ms): 527.28 
iteration: 42 | loss: 8.867 | iteration time (ms): 523.56 
iteration: 43 | loss: 8.705 | iteration time (ms): 524.88 
iteration: 44 | loss: 8.815 | iteration time (ms): 523.07 
iteration: 45 | loss: 8.655 | iteration time (ms): 525.73 
iteration: 46 | loss: 8.740 | iteration time (ms): 525.80 
iteration: 47 | loss: 8.821 | iteration time (ms): 523.97 
iteration: 48 | loss: 8.625 | iteration time (ms): 524.56 
iteration: 49 | loss: 8.520 | iteration time (ms): 524.56 
iteration: 50 | loss: 8.488 | iteration time (ms): 521.91 
...
```
### Running on AMD GPUs

To run on AMD hardware, you must comment out lines 144–162 in the `initialize.py` file within the Megatron submodule. These lines attempt to locate the `nvcc` compiler, which is not available in AMD environments. This change does not impact performance, as fused kernels are not loaded from this location in current implementations.



## Build Apex from source
```
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key...
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" --config-settings "--build-option=--fast_layer_norm" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" --config-settings "--build-option=--fast_layer_norm" ./
```