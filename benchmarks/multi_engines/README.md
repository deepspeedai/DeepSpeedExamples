# Multi-engine process-group checks

Small `torch.distributed` checks behind the design discussion in
[DeepSpeed RFC #9081](https://github.com/deepspeedai/DeepSpeed/issues/9081): several DeepSpeed engines with
different parallel layouts in one process. They test the process-group behavior that design depends on
(how groups are created on subsets of ranks, how they are destroyed, what they cost, and what happens when a
collective times out) using plain `torch.distributed`, without DeepSpeed.

The scripts are device-agnostic. They pick the current torch accelerator (CUDA, XPU, HPU, NPU, ...) and torch's
default backend for it (NCCL, XCCL, HCCL, ...), or CPU with Gloo, and both can be overridden.

## Requirements

- One node with **4 devices** (the `timeout_probe` case uses 2).
- PyTorch with `torch.distributed` for your backend. `split_group` and `shrink_group` cases are skipped on
  torch versions without them.

## Running

```bash
cd benchmarks/multi_engines
bash run_checks.sh                               # all cases, default device/backend, eager init
EAGER=0 bash run_checks.sh                       # lazy init (no device_id at init_process_group)
BACKEND=gloo DEVICE=cpu bash run_checks.sh       # CPU with Gloo
CASES="registry_path destroy_diff_order" REPEATS=3 bash run_checks.sh
python summarize.py logs/<timestamp>             # re-summarize a log directory
```

One case directly:

```bash
CASE=registry_path torchrun --nproc_per_node=4 multi_engine_checks.py
```

| Variable | Meaning | Default |
|---|---|---|
| `DEVICE` | Accelerator type (`cuda`, `xpu`, `hpu`, `npu`, `cpu`, ...) | the current torch accelerator, else `cpu` |
| `BACKEND` | `torch.distributed` backend | torch's default for `DEVICE` |
| `EAGER` | `1` binds the device at `init_process_group(device_id=...)`; `0` leaves init lazy | `1` |
| `CASES`, `REPEATS` | Cases to run and how many times | all cases, 1 |
| `CASE_TIMEOUT` | Seconds before a case is declared hung | 150 |
| `PYTHON`, `LOG_DIR`, `MASTER_PORT` | Interpreter, log directory, first port | `python`, `logs/<timestamp>`, 29650 |

Cases expected to hang are bounded by `CASE_TIMEOUT`, so a full run takes about 20-30 minutes. On torch 2.15 with
NCCL, `TORCH_DIST_USE_NCCL2=0` selects the legacy `ProcessGroupNCCL` instead of the default `ProcessGroupNCCL2`.

## Cases

Every case runs in its own process and checks each group with a size all-reduce and a rank-sum all-reduce
against the expected values.

| Case | What it does | What the design needs |
|---|---|---|
| `registry_path` | The proposed scheme. Every rank creates every group of every engine, in the same order, and keeps the handle only where it is a member. Each engine has its own root group, and release is deferred to the next all-ranks call. Engines A (TP2 × DP2) and B (TP4) used alternately, then DP4 → DP2 → DP4 with the DP2 engine on ranks 0-1 only | pass |
| `destroy_same_order` | Two groups over ranks 0-1 destroyed in the same order on both | pass |
| `memory_cycles` | 5 create/use/destroy cycles of a full-world group, reporting free device memory | pass, with memory returned |
| `overlap_members_only` | Overlapping groups [0,1,2] and [1,2,3] created only by their members (`use_local_synchronization=True`), in one order | hazard: expected to hang or return wrong values |
| `subset_then_all` | A members-only group [0,1,2], then an all-ranks group | hazard |
| `split_then_all` | A `split_group` called only by the members of its parent, then an all-ranks group; group names are compared across ranks | hazard: values may be right while names differ |
| `destroy_diff_order` | The same two groups destroyed in different orders on the two member ranks | hazard |
| `shrink_then_all` | `shrink_group` excluding rank 3 (rank 3 waits until the others finish), with and without a later all-ranks group | unknown |
| `timeout_probe` | 2 ranks; rank 1 never joins a collective. Records what the call returns, the backend's error state and the cost of querying it, and how long the process stays alive | observe |

## Results so far

Single node, 4× NVIDIA B200 over NVLink:

| Case | torch 2.15 / ProcessGroupNCCL2 | torch 2.15 / legacy ProcessGroupNCCL | torch 2.9.1 / ProcessGroupNCCL |
|---|---|---|---|
| `registry_path` | pass | pass | pass (eager and lazy) |
| `destroy_same_order` | pass | not run | pass |
| `memory_cycles` | pass; about 580 MiB per GPU per communicator with NVLS off, 630-646 MiB with NVLS on, all returned on destroy | same | same |
| `overlap_members_only` | wrong values, then hang | lazy: names differ across ranks, hang; eager: wrong values, then hang | lazy: hang; eager: wrong values, then hang |
| `subset_then_all` | wrong values, then hang | lazy: names differ across ranks, hang; eager: wrong values, then hang | not run |
| `split_then_all` | values correct, names differ across ranks | values correct, names differ across ranks | not run |
| `destroy_diff_order` | hang | not run | hang |
| `shrink_then_all` | fail | fail | not run |
| `timeout_probe` | call returns a wrong value after about 16 s without raising; error state `TIMEOUT`; process aborted about 15 s later | call blocks about 75 s, returns a wrong value; error state `TIMEOUT`; killed 1-3 s later | same as legacy |

Gloo on CPU:
- torch 2.9.1: `registry_path`, `destroy_same_order`, and `memory_cycles` pass. `split_then_all` raises
  `No device associated with the default pg, not safe to split any process groups`. In `timeout_probe` the
  timed-out call **raises** after 15 s (unlike NCCL), the process stays alive, and Gloo exposes no error state.
- torch 2.15: `split_then_all` hangs after the members-only split.

XPU, HCCL, and multi-node have not been run. Results from other
accelerators and backends are very welcome on the RFC thread: please include the `summarize.py` output and the
torch, backend, and driver versions.

## Files

- `multi_engine_checks.py`: the cases.
- `common.py`: device and backend selection, result records, all-reduce checks.
- `run_checks.sh`: runs every case with `torchrun` and summarizes.
- `summarize.py`: one line per run, compared with what the design needs from that case.
