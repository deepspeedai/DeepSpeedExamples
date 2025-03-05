#### Install

``pip install torch torchvision``
You will also need to install updated DeepSpeed version (>0.7.0), which contains the compression library.

#### Key File: train.py

The python code is modified based on (https://github.com/deepspeedai/DeepSpeedExamples/tree/master/cifar). The key added feature is the compression pipeline.

#### Folders (config)

* **config:** This folder provides DeepSpeed configuration, including quantization, pruning and layer reduction.

#### bash script 
* **run_compress.sh**  This bash script contains jobs for training a checkpoint and then compressing this checkpoint.  See more descriptions and results in our [tutorial page](https://www.deepspeed.ai/).

