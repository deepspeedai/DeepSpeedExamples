# Not maintained / deprecated

> __Warning__
> all future/current changes are now in new [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed). 

### Notes on 3 deprecated Megatron folders in this repository

Megatron-LM : This is a fairly old snapshot of Megatron-LM , and we have been using it show case the earlier features of DeepSpeed. This does not contain ZeRO-3 or 3D parallelism.

Megatron-LM-v1.1.5-3D_parallelism: This is a relatively new Megatron (Oct 2020), but before Megatron started supporting 3D parallelism. We ported this version to showcase how to use 3D parallelism inside DeepSpeed with Megatron.

Megatron-LM-v1.1.5-ZeRO3: The underlying Megatron version is same as the 3D_parallelism but it does not contain the 3D parallelism port. It however contains the most recent advances in DeepSpeed including ZeRO-3, ZeRO-3 Offload and ZeRO-Infinity. We did this separately from 3D parallelism port to isolate the changes required for each of them and to avoid users combining them together which is not supported, and will likely lead to more confusion. 

3D parallelism is quite similar in both DeepSpeed and new Megatron, we don't have plans to support their combination. The Megatron-DeepSpeed repository supports DeepSpeed's 3D parallelism (pipeline-parallelism inside DeepSpeed and megatron/mpu-based tensor-parallelism).
