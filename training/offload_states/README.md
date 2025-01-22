# Offloading States Example

The script `offload_states.py` demonstrates how to offload the state of a model. Here is the example usage.

```bash
$ deepspeed --num_gpus=4 offload_states.py --hidden_dim 32768 --nlayers 4 --pin_memory --non_blocking
...
Memory usage (0): include=None, pin_memory=True, non_blocking=True alloc_before_offload=18198419456 alloc_after_offload=17763840
Memory usage (1): include=None, pin_memory=True, non_blocking=True alloc_before_offload=18198760960 alloc_after_offload=17763840
...
Summary: pin_memory=True non_blocking=True offload=5.643414640426636 load=2.4087101459503173
```

`run_benchmark.sh` shows how to run the script with different configurations. The script outputs the time for offloading and loading the states.

```bash
$ ./run_benchmark.sh
...
|   |pin_memory=0_non_blocking=0|pin_memory=0_non_blocking=1|pin_memory=1_non_blocking=0|pin_memory=1_non_blocking=1|
|--:|---------------------------|---------------------------|---------------------------|---------------------------|
|  1|4.34 / 3.42                |4.99 / 2.37                |6.5 / 2.42                 |6.0 / 2.39                 |
|  2|9.9 / 3.28                 |5.1 / 2.34                 |6.21 / 2.42                |6.25 / 2.45                |
|  3|9.92 / 3.19                |6.71 / 2.35                |6.33 / 2.38                |5.93 / 2.42                |
|  4|9.55 / 2.82                |7.11 / 2.39                |6.9 / 2.38                 |6.5 / 2.43                 |...
```
