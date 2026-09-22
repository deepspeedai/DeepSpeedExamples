# ZeRO-3 CPU-Offload Pinned-Memory Benchmark

This directory contains an end-to-end benchmark for ZeRO-3 CPU offload that
measures training step time with the native pinned host-memory backend,
comparing device-registered vs unregistered pinned memory
(`DS_PIN_MEMORY_REGISTER_DEVICE=1` vs `0`).

The benchmark runs ZeRO-3 with `offload_optimizer` and `offload_param` (both
CPU, `pin_memory=True`) and `DS_PIN_MEMORY_BACKEND=native`, on any accelerator
with native pin + `register_host_memory` support (CUDA and XPU are tested).
Each arm runs in its own subprocess with a fresh rendezvous port so device
state never leaks between arms.

## Files in this Directory

- **zero3_offload_bench.py**: Benchmarking script; the model can be a real
  architecture fetched from the HuggingFace hub (random weights) or a
  synthetic MLP stack that needs no network access.

## Usage

```bash
# real model architecture (config fetched from the HF hub, random weights)
python zero3_offload_bench.py --model Qwen/Qwen2.5-7B --batch 4 --seq 512

# synthetic MLP stack, no network needed
python zero3_offload_bench.py --hidden 2048 --layers 12 --batch 4 --seq 128
```

Results are printed as JSON with per-arm step times.

> **Note**: the native backend mlocks host memory; raise `RLIMIT_MEMLOCK`
> (`ulimit -l`) or run as root for multi-GB models.
