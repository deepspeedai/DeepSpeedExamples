# ZenFlow Benchmark Example


Please install DeepSpeed via pip install deepspeed if you haven't already done so. 

```bash
pip install -r requirements.txt
```


The script `zf_benchmark.py ` demonstrates how to offload the state of a model. Here is the example usage.

```python
$ deepspeed --num_gpus=4 zf_benchmark.py --hidden_dim 4096 --nlayers 4 --iteration 5 --pin_memory_opts 1 --topk_ratios 0.1 --update_intervals 2 --overlap_steps
...
time (ms) | selective_optimizer_update: 19.20 | selective_optimizer_process: 28.80 | selective_optimizer_sync: 0.05
time (ms) | fwd_microstep: 54.76 | bwd_microstep: 122.95 | bwd_inner_microstep: 12.22 | bwd_allreduce_microstep: 103.64 | step_microstep: 0.34
Step 0 time: 178.66ms
time (ms) | optimizer_allgather: 26.19 | optimizer_gradients: 26.06 | optimizer_step: 128.20
time (ms) | selective_optimizer_update: 0.00 | selective_optimizer_process: 0.57 | selective_optimizer_step: 1.48 | selective_optimizer_sync: 0.00
time (ms) | fwd_microstep: 0.38 | bwd_microstep: 57.88 | bwd_inner_microstep: 1.06 | bwd_allreduce_microstep: 56.50 | step_microstep: 183.27
time (ms) | fwd: 55.15 | bwd: 180.82 | bwd_inner: 13.28 | bwd_allreduce: 160.15 | step: 183.61
Step 1 time: 242.16ms
time (ms) | selective_optimizer_update: 0.00 | selective_optimizer_process: 1.58 | selective_optimizer_step: 0.00 | selective_optimizer_sync: 0.00
time (ms) | fwd_microstep: 0.30 | bwd_microstep: 16.73 | bwd_inner_microstep: 1.39 | bwd_allreduce_microstep: 14.96 | step_microstep: 0.20
Step 2 time: 17.60ms
time (ms) | optimizer_allgather: 0.65 | optimizer_gradients: 16.95 | optimizer_step: 108.45
time (ms) | selective_optimizer_update: 0.00 | selective_optimizer_process: 0.56 | selective_optimizer_step: 1.42 | selective_optimizer_sync: 0.00
time (ms) | fwd_microstep: 0.29 | bwd_microstep: 36.65 | bwd_inner_microstep: 0.95 | bwd_allreduce_microstep: 35.51 | step_microstep: 128.57
time (ms) | fwd: 0.59 | bwd: 53.39 | bwd_inner: 2.33 | bwd_allreduce: 50.48 | step: 128.77
Step 3 time: 166.10ms
time (ms) | selective_optimizer_update: 0.00 | selective_optimizer_process: 1.57 | selective_optimizer_step: 0.00 | selective_optimizer_sync: 0.00
time (ms) | fwd_microstep: 0.31 | bwd_microstep: 15.47 | bwd_inner_microstep: 1.33 | bwd_allreduce_microstep: 13.97 | step_microstep: 0.23
...
[Summary] pin_memory=False topk_ratio=0.1 update_interval=2 overlap_step=False avg_accumulation_step=16.77ms avg_update_step=171.38ms
```

`run_benchmark.sh` shows how to run the script with different configurations. The script outputs the time for offloading and loading the states.

```python
$ ./run_benchmark.sh
...
+---------+--------------+-------------------+----------------+--------------+-----------------+----------------+----------------+-------------------------------------+
|   trial |   topk_ratio |   update_interval | overlap_step   | pin_memory   |   avg_step (ms) |   avg_bwd (ms) |   avg_fwd (ms) |   avg_selective_optimizer_step (ms) |
|---------+--------------+-------------------+----------------+--------------+-----------------+----------------+----------------+-------------------------------------|
|       1 |          0.1 |                 2 | False          | False        |         24.0153 |        12.8377 |        1.91733 |                            0.247    |
|       1 |          0.1 |                 2 | False          | False        |         22.8293 |        12.5187 |        1.73767 |                            0.258333 |
|       1 |          0.1 |                 2 | False          | True         |         21.6523 |        10.2863 |        1.97767 |                            0.250333 |
|       1 |          0.1 |                 4 | False          | False        |         14.2108 |        10.9072 |        1.2436  |                            0.1484   |
|       1 |          0.1 |                 4 | False          | False        |         13.6408 |        10.8386 |        1.2208  |                            0.1456   |
|       1 |          0.1 |                 4 | False          | True         |         12.863  |         9.0592 |        1.2148  |                            0.1464   |...
```


**Notes:** Each row in the table represents the average performance metrics for a specific configuration of ZenFlow’s offloading setup, defined by:

- **`topk_ratio`**: The fraction of parameters selected for offloading during each update.
- **`update_interval`**: How often (in steps) the offloading state is updated.
- **`overlap_step`**: Whether overlapping offloading with computation is enabled.
- **`pin_memory`**: Whether pinned host memory is used to speed up data transfer between CPU and GPU.

The performance metrics include:

- **`avg_step (ms)`**: Total time per training step — the primary measure of end-to-end training performance.
- **`avg_bwd (ms)`**: Time spent in the backward pass, including gradient computation and allreduce.
- **`avg_fwd (ms)`**: Time spent in the forward pass.
- **`avg_selective_optimizer_step (ms)`**: Time spent in the selective optimizer step — indicates overhead introduced by ZenFlow’s offloading logic.

**Tips for Analysis:**

- Lower **`avg_step`** means faster training.
- Comparing configurations helps identify performance trade-offs (e.g., `pin_memory=True` often reduces transfer latency).
- A higher **`update_interval`** typically reduces offloading frequency and overhead.
- Enabling **`overlap_step=True`** can further hide offloading latency behind computation when the model update phase is longer.