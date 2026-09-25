# ZeRO-3 CPU-Offload Pinned-Memory Benchmark

This directory contains an end-to-end benchmark for ZeRO-3 CPU offload that
measures training step time with pinned vs unpinned host memory, plus an
opt-in ablation of registered vs unregistered pinned memory.

## Files in this Directory

- **zero3_offload_bench.py**: Benchmarking script; the model can be a real
  architecture fetched from the HuggingFace hub (random weights) or a
  synthetic MLP stack that needs no network access.

## What it Measures

By default the script runs ZeRO-3 with `offload_optimizer` and `offload_param`
(both CPU) in two arms and reports the step-time comparison:

| Arm | offload `pin_memory` | `DS_PIN_MEMORY_REGISTER_DEVICE` |
|-----|----------------------|---------------------------------|
| `unpinned` | `False` | (n/a) |
| `pinned` | `True` (`DS_PIN_MEMORY_BACKEND=native`) | `1` |

Works on any accelerator with native pin + `register_host_memory` support
(CUDA and XPU are tested). Each arm runs in its own subprocess with a fresh
rendezvous port so device state never leaks between arms.

Power users can additionally ablate device registration of pinned buffers:

```bash
python zero3_offload_bench.py --ablate-register ...
```

which adds a `pinned-unregistered` arm (`DS_PIN_MEMORY_REGISTER_DEVICE=0`).

## Usage

```bash
# real model architecture (config fetched from the HF hub, random weights)
python zero3_offload_bench.py --model Qwen/Qwen2.5-7B --batch 4 --seq 512

# synthetic MLP stack, no network needed
python zero3_offload_bench.py --hidden 2048 --layers 12 --batch 4 --seq 128
```

Results are printed as a table (avg/min step time, GPU peak memory) and as a
JSON line (`DRIVERRESULT=...`) with per-arm details and the pinning speedup.

> **Note**: the native backend mlocks host memory; raise `RLIMIT_MEMLOCK`
> (`ulimit -l`) or run as root for multi-GB models.
