# Pinned-memory experiments

Harnesses for pin vs pageable host memory on CPU offload. Each subdirectory is
one experiment. Shared subprocess/JSON helpers live in `common.py`.

Native backends `mlock` host memory: raise `RLIMIT_MEMLOCK` (`ulimit -l`) or
run as root for multi-GB models.

## Layout

| Folder | Blog experiment | Default command |
|--------|-----------------|-----------------|
| [`model_tensor_offload/`](model_tensor_offload/) | ZeRO CPU param/optimizer offload (stage 3 default; `--zero-stage 1\|2` optional) | `python model_tensor_offload/bench.py --hidden 2048 --layers 12 --batch 4 --seq 128` |
| [`activation_offload/`](activation_offload/) | Checkpoint hidden-state offload; `use_pin_memory` on/off, **async on** | `python activation_offload/bench.py --hidden 1024 --layers 8 --batch 1 --seq 2048` |
| [`h2d_d2h/`](h2d_d2h/) | Supporting H2D/D2H GB/s (pageable, torch, native-unregistered, native-registered) | `python h2d_d2h/bench.py` |
| [`grad_offload/`](grad_offload/) | Optional #8207-style grad offload (wraps model-tensor ZeRO-3) | `python grad_offload/bench.py --hidden 2048 --layers 12` |
| [`cpu_pin/`](cpu_pin/) | Optional CPU-only native vs Torch pin | `python cpu_pin/bench.py` |
| [`deepcompile_activation/`](deepcompile_activation/) | Optional `compile.offload_activation_pin_memory` | `python deepcompile_activation/bench.py` |

`zero3_offload_bench.py` at this directory root still runs **model-tensor ZeRO-3** (same flags as before).

## Model-tensor arms

| Arm | `offload_*.pin_memory` | Backend |
|-----|------------------------|---------|
| unpinned | `false` | n/a |
| pinned | `true` | `DS_PIN_MEMORY_BACKEND=native`, `DS_PIN_MEMORY_REGISTER_DEVICE=1` |

`--ablate-register` adds `pinned-unregistered`. CUDA-oriented; skip on XPU if `register_host_memory` is missing (`h2d_d2h/bench.py --skip-native-register`).

```bash
python model_tensor_offload/bench.py --model Qwen/Qwen2.5-7B --batch 4 --seq 512
python model_tensor_offload/bench.py --zero-stage 2 --hidden 2048 --layers 12 --batch 4 --seq 128
```

Each arm is a subprocess with a fresh rendezvous port. Results: table plus `DRIVERRESULT=` JSON.
