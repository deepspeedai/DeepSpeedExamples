# Domino Example

## Install Dependency Libraries
```
pip install -r requirements.txt
```

## Prepare the Dataset
Follow the instructions from [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/tree/main/examples_deepspeed/universal_checkpointing#download-and-pre-process-training-dataset) to prepare the training dataset.

## Execute Domino Training

To start training, adjust the following parameters in the script as needed:

- **GPUS_PER_NODE**: Number of GPUs per node.
- **CHECKPOINT_PATH**: Path to the checkpoint, if applicable.
- **VOCAB_FILE**, **MERGE_FILE**, **DATA_PATH**: Paths to the dataset files.
- **--micro-batch-size**: Batch size per GPU.

### Available Models and Scripts

| Model      | Script                   |
|------------|--------------------------|
| GPT-3 2.7B | `pretrain_gpt3_2.7b.sh`  |
| GPT-3 6.7B | `pretrain_gpt3_6.7b.sh`  |
| LLaMA 7B   | `pretrain_llama_7b.sh`   |
| LLaMA 13B  | `pretrain_llama_13b.sh`  |

### Example

To train the GPT-3 2.7B model, run the following command:

```bash
bash pretrain_gpt3_2.7b.sh
```

The output should look like this:

```
training ...
iteration: 1 | loss: 11.318 | iteration time (ms): 2174.0469932556152 
iteration: 2 | loss: 11.307 | iteration time (ms): 1414.4024848937988 
iteration: 3 | loss: 11.323 | iteration time (ms): 1385.9455585479736 
iteration: 4 | loss: 11.310 | iteration time (ms): 1475.5175113677979 
iteration: 5 | loss: 11.306 | iteration time (ms): 1395.7207202911377 
iteration: 6 | loss: 11.315 | iteration time (ms): 1392.2104835510254 
iteration: 7 | loss: 11.314 | iteration time (ms): 1402.6703834533691 
iteration: 8 | loss: 11.309 | iteration time (ms): 1450.613260269165 
iteration: 9 | loss: 11.305 | iteration time (ms): 1473.1688499450684 
iteration: 10 | loss: 11.320 | iteration time (ms): 1398.4534740447998 
[2024-11-04 15:32:30,918] [INFO] [launch.py:351:main] Process 73015 exits successfully.
[2024-11-04 15:32:30,918] [INFO] [launch.py:351:main] Process 73017 exits successfully.
[2024-11-04 15:32:30,919] [INFO] [launch.py:351:main] Process 73014 exits successfully.
[2024-11-04 15:32:30,919] [INFO] [launch.py:351:main] Process 73016 exits successfully.
```

## Advanced Usage
You can compile Pytorch and Apex from source for better performance.

### Compile PyTorch from Source
Compile PyTorch from source could enable JIT script.
```
git clone -b v2.1.0 https://github.com/pytorch/pytorch.git
git submodule sync
git submodule update --init --recursive
conda install cmake ninja
pip install -r requirements.txt
conda install intel::mkl-static intel::mkl-include
conda install -c pytorch magma-cuda121 # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py develop

# Build torchvision
git clone https://github.com/pytorch/vision.git
python setup.py develop
```

## Build Apex
```
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key...
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" --config-settings "--build-option=--fast_layer_norm" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" --config-settings "--build-option=--fast_layer_norm" ./
```