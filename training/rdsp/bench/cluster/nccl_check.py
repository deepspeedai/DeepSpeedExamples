# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""All-reduce and all-to-all bandwidth over every GPU of both nodes, with NCCL set up as the
runs use it (EFA through the aws-ofi-nccl plugin, bench/cluster/efa_env.sh). Start on both
nodes at once:

    bash bench/cluster/both.sh bash bench/cluster/nccl_check.sh

Rank 0 prints each size's time and bus bandwidth (2 (n-1)/n x bytes / time). Over TCP instead of
EFA the large sizes come out an order of magnitude lower."""

import os
import time

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
for mib in (16, 256, 1024):
    x = torch.ones(mib * 2**20 // 2, dtype=torch.bfloat16, device="cuda")
    for _ in range(5):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    start, n = time.perf_counter(), 20
    for _ in range(n):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / n
    bus = 2 * (world - 1) / world * x.numel() * x.element_size() / seconds / 1e9
    if rank == 0:
        print(
            f"all_reduce {mib} MiB over {world} GPUs: {seconds * 1e3:.2f} ms, "
            f"bus bandwidth {bus:.1f} GB/s",
            flush=True,
        )
# all-to-all, as expert parallelism sends tokens: each rank sends 1/world of the buffer to every
# rank. Bus bandwidth (n-1)/n x bytes / time, as nccl-tests reports it.
for mib in (16, 256, 1024):
    x = torch.ones(mib * 2**20 // 2, dtype=torch.bfloat16, device="cuda")
    y = torch.empty_like(x)
    for _ in range(5):
        dist.all_to_all_single(y, x)
    torch.cuda.synchronize()
    start, n = time.perf_counter(), 20
    for _ in range(n):
        dist.all_to_all_single(y, x)
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / n
    bus = (world - 1) / world * x.numel() * x.element_size() / seconds / 1e9
    if rank == 0:
        print(
            f"all_to_all {mib} MiB per GPU over {world} GPUs: {seconds * 1e3:.2f} ms, "
            f"bus bandwidth {bus:.1f} GB/s",
            flush=True,
        )
dist.destroy_process_group()
