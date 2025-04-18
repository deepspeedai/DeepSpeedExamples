# SGLang + ZeRO-Inference Examples
This folder contains examples of [ZeRO-Inference](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/inference/huggingface/zero_inference/README.md) integration into [SGLang](https://github.com/sgl-project/sglang) framework. This integration enable SGLang to inference massive models (e.g., with 100s billion parameters) on a single GPU through the NVMe/CPU offloading optimizations of ZeRO-Inference. 

## Prerequisites
1. DeepSpeed version >= [0.16.6](https://github.com/deepspeedai/DeepSpeed/releases/tag/v0.16.6)
2. SGLang: These examples require our SGLang [fork](https://github.com/tjruwase/sglang/tree/zero-inference). We plan to upstream the SGLang changes to main branch. 


## Examples
The examples comprise of the following:
1. bash scripts that benchmark SGLang throughput in [offline mode](https://github.com/sgl-project/sglang/blob/main/python/sglang/bench_offline_throughput.py) with different ZeRO-Inference offloading options. Each script runs a inference on a different model with a prompt of 512 tokens, output of 32 tokens, and batch size of 128. 
2. DeepSpeed config files corresponding to ZeRO-Inference offloading: (i) CPU offload, (ii) NVMe offload with AIO, and (iii) NVMe offloading with NVIDIA GDS. 