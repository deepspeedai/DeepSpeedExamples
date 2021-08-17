Thanks Gopi Kumar for contributing this example, demonstrating how to apply DeepSpeed to CIFAR-10 model.

cifar10_tutorial.py
    Baseline CIFAR-10 model.

cifar10_deepspeed.py
    DeepSpeed applied CIFAR-10 model.

ds_config.json
    DeepSpeed configuration file.

run_ds.sh
    Script for running DeepSpeed applied model.

run_ds_moe.sh
    Script for running DeepSpeed model with Mixture of Experts (MoE) integration.

* To run baseline CIFAR-10 model - "python cifar10_tutorial.py"
* To run DeepSpeed CIFAR-10 model - "bash run_ds.sh"
* To run DeepSpeed CIFAR-10 model with Mixture of Experts (MoE) - "bash run_ds_moe.sh"
