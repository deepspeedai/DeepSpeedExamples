# Using DeepNVMe for simple file reads and writes involving CPU/GPU tensors

The purpose of this folder is to provide example codes that illustrate how to use [DeepNVMe](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-gds/README.md) for simple file operations of moving raw data bytes between persistent storage and CPU/GPU tensors. For each file operation, we provide an implementation using Python I/O functionality, and a DeepNVMe implementation using CPU bounce buffer (aio) and NVIDIA Magnum IO<sup>TM</sup> GPUDirect® Storage (GDS) as appropriate. 

The following table is a mapping of file operations to the corresponding Python and DeepNVMe implementations. 


File Operation | Python | DeepNVMe (aio) | DeepNVMe (GDS)
|---|---|---|---|
Load CPU tensor from file | py_load_cpu_tensor.py | aio_load_cpu_tensor.py | - |
Load GPU tensor from file | py_load_gpu_tensor.py | aio_load_gpu_tensor.py | gds_load_gpu_tensor.py |
Store CPU tensor to file | py_store_cpu_tensor.py | aio_store_cpu_tensor.py | - |
Store GPU tensor to file | py_store_gpu_tensor.py | aio_store_gpu_tensor.py | gds_store_gpu_tensor.py |  

The Python implementations are the scripts with `py_` prefix. while the DeepNVMe implementations are those with`aio_` and `gds_`prefixes. 

## Requirements 
Ensure your environment is properly configured to run these examples. First, you need to install DeepSpeed version >= 0.15.0. Next, ensure that the DeepNVMe operators are available in the DeepSpeed installation. The `async_io` operator is required for any DeepNVMe functionality, while the `gds` operator is required only for GDS functionality. You can confirm availability of each operator by inspecting the output of `ds_report` to check that compatible status is <span style="color:green">[OKAY]</span>. Below is a snippet of `ds_report` output showing availability of both `async_io` and `gds` operators. 

<div align="center">
    <img src="./media/deepnvme_ops_report.png" style="width:6.5in;height:3.42153in" />
</div> 
<div align="center">
    ds_report output showing availability of DeepNVMe operators (async_io and gds) in a DeepSpeed installation. 
</div> 


If `async_io` opertator is unavailable, you will need to install the appropriate `libaio` library binaries for your Linux flavor. For example, Ubuntu users will need to run `apt install libaio-dev`. In general, you should carefully inspect `ds_report` output for helpful tips such as the following: 

```bash
[WARNING]  async_io requires the dev libaio .so object and headers but these were not found.
[WARNING]  async_io: please install the libaio-dev package with apt
[WARNING]  If libaio is already installed (perhaps from source), try setting the CFLAGS and LDFLAGS environment variables to where it can be found.
```

To enable `gds` operator, you will need to install NVIDIA GDS by consulting the appropriate guide for [bare-metal systems](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html) or Azure VMs (coming soon). 

## Tensor Load Examples
The tensor load example scripts share a common command-line interface, which is illustrated below using `py_read_load_cpu_tensor.py`.
```bash
$ python py_load_cpu_tensor.py --help
usage: py_load_cpu_tensor.py [-h] --input_file INPUT_FILE [--loop LOOP] [--validate]

options:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        File on NVMe device that will read as input.
  --loop LOOP           The number of times to repeat the operation (default 3).
  --validate            Run validation step that compares tensor value against Python file read
```
Before running these example scripts ensure that the input file exists on an NVMe device. The `--validate` option is relevant only to the DeepNVme implementations. This option provides minimal correctness checking by comparing against a tensor loaded using Python. We also provide a bash script `run_load_tensor.sh`, which runs all the example tensor load scripts.


## Tensor Store Examples
The tensor store examples share a command-line interface, which is illustrated below using `py_store_cpu_tensor.py`
```bash
$ python py_store_cpu_tensor.py --help
usage: py_store_cpu_tensor.py [-h] --nvme_folder NVME_FOLDER [--mb_size MB_SIZE] [--loop LOOP] [--validate]

options:
  -h, --help            show this help message and exit
  --nvme_folder NVME_FOLDER
                        NVMe folder for file write.
  --mb_size MB_SIZE     Size of tensor to save in MB (default 1024).
  --loop LOOP           The number of times to repeat the operation (default 3).
  --validate            Run validation step that compares tensor value against Python file read

```
Before running these examples ensure that the output folder exists on an NVMe device and that you have write permission. The `--validate` option is relevant only to the DeepNVMe implementations. This option provides minimal correctness checking by comparing the output file against that created using Python. We also provide a bash script `run_store_tensor.sh`, which runs all the example tensor store scripts.  


## Performance Advisory
Although this folder is primarily meant to help with integrating DeepNVMe into your Deep Learning applications, the example scripts also print out performance numbers of read and write throughput. So, we expect you will observe some performance advantage of DeepNVMe compared to Python. However, do note that it is likely that better performance can be realized by tuning DeepNVMe for your environment. Such tuning efforts will ideally generate more optimal values for configuring DeepNVMe. 

For reference, DeepNVMe configuration using hard-coded constants for `aio_` implementations is as follows:

```python
    aio_handle = AsyncIOBuilder().load().aio_handle(1024**2, 128, True, True, 1)
```

The corresponding DeepNVMe configuration for `gds_` implementations is as follows:

```python
    gds_handle = GDSBuilder().load().gds_handle(1024**2, 128, True, True, 1)
```

Despite the above caveat, it seems that some performance numbers would be useful here to help set the right expectations. The experiments were conducted on an Azure [NC80adis_H100_v5](https://learn.microsoft.com/en-us/azure/virtual-machines/ncads-h100-v5) series virtual machine (VM). This VM includes two 3.5TB local NVMe devices (labelled Microsoft NVMe Direct Disk v2) that we combined into a single RAID-0 volume. The software environment included Ubuntu 22.04.4 LTS, Linux kernel 6.5.0-26-generic, Pytorch 2.4, and CUDA 12.4. We ran experiments of 1GB data transfers using the unmodified scripts, i.e., without DeepNVMe tuning, and present the throughput results in the tables below. In summary, we observed that DeepNVMe significantly accelerates I/O operations compared to Python. DeepNVMe is 8-16X faster for loading tensor data, and 11X-119X faster for writing tensor data. 

Load 1GB CPU tensor (1GB file read) | GB/sec | Speedup over Python | 
|---|---|---|
py_load_cpu_tensor.py  | 1.5 | - | 
aio_load_cpu_tensor.py | 12.3 | 8X | 

Load 1GB GPU tensor (1GB file read) | GB/sec | Speedup over Python | 
|---|---|---|
py_load_gpu_tensor.py | 0.7| - | 
aio_load_gpu_tensor.py | 9.9 | 14X | 
gds_load_gpu_tensor.py | 11.1 | 16X | 


Store 1GB CPU tensor (1GB file write) | GB/sec | Speedup over Python | 
|---|---|---|
py_store_cpu_tensor.py  | 0.7 | - | 
aio_store_cpu_tensor.py | 8.1 | 11X | 


Store 1GB GPU tensor (1GB file write) | GB/sec | Speedup over Python | 
|---|---|---|
py_store_gpu_tensor.py | 0.5 | - | 
aio_store_gpu_tensor.py | 8.3 | 18X | 
gds_store_gpu_tensor.py | 8.6 | 19X | 



# Conclusion
We hope you find this document and example scripts useful for integrating DeepNVMe into your applications. 
