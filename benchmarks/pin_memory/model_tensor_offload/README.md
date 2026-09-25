# Model-tensor CPU offload

Pin vs pageable step time for ZeRO CPU offload of parameters and/or optimizer
states. Default `--zero-stage 3` (param + optimizer). `--zero-stage 1` or `2`
offloads the optimizer only.

See the [parent README](../README.md).
