[FastPersist](https://arxiv.org/abs/2406.13768) is an optimization technique that leverages NVMe storage to accelerate model checkpointing. This folder contains micro-benchmarks and instructions for demonstrating FastPersist. 

## Enabling FastPersist Optimizations ##
FastPersist is designed to integrate with torch checkpointing and has been validated with torch version 2.6.0. This requires slight modifications to torch serialization, and for convenience we provide [original](torch/serialization_orig_v2.6.0.py) and [patched](torch/serialization_fast_v2.6.0.py) versions of serialization.py. Thus, to demonstrate FastPersist performance you need to overwrite `torch/serialization.py` in your torch installation with the patched version. 

## Available Micro-benchmarks ##
This folder contains three different micro-benchmarks that are implemented by the following scripts:
1. torch_save_tensor.py: Serialize a raw pytorch tensor to disk using `torch.save()` integration.
2. torch_save_model.py: Serialize a HF model to disk using `torch.save()` integration. 
3. deepspeed_save_model.py: Serialize a HF model to disk using DeepSped integration. 

Each script provides a `--help` option to examine the available configurations. The scripts are written for single-process execution and so can be launched using `python`. 

As an example, the performance of using the `torch.save()` integration of checkpointing HF phi-3-mini model from GPU memory can be measured as follows: 
```
python torch_save_model.py --model phi3 --folder /mnt/nvme0 --gpu
```

The script executes and reports the performance of the checkpointing workload using different mechanisms including vanilla `torch.save()`, FastPersist with CPU bounce buffer, FastPersist with NVIDIA GDS, etc. You can find the respective performance by searching the generated log for lines similar to the following snippet. For this example, the results below, collected using eight PCI Gen4 NVMes RAID-0 (data-striped), show checkpointing throughputs of 0.69GB/sec and 17.75GB/sec for vanilla `torch.save()` (labelled test_save) and FastPersist with CPU bounce buffer (labelled test_ds_aio_fast_save) respectively. 

```bash
test_save -- 14.23 GB, 20.72 secs,  0.69 GB/s
test_ds_aio_fast_save -- 14.23 GB,  0.80 secs, 17.75 GB/s
```

