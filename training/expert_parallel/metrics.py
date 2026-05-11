"""Minimal CSV metrics and distributed reduction helpers for the AutoEP example."""

from __future__ import annotations

import csv
import os
from typing import Any

import torch
import torch.distributed as dist

METRICS_COLUMNS: list[str] = [
    "step",
    "loss",
    "iter_time_sec",
    "global_tokens_per_sec",
    "cuda_memory_allocated_bytes",
    "cuda_peak_memory_allocated_bytes",
    "cuda_peak_memory_reserved_bytes",
]


class MetricsLogger:
    """Write per-step metrics to CSV on rank 0."""

    def __init__(self, csv_path: str, rank: int) -> None:
        self.csv_path = csv_path
        self.rank = rank
        self._file = None
        self._writer = None

    def log_step(self, metrics: dict[str, Any]) -> None:
        if self.rank != 0:
            return
        if self._file is None:
            parent = os.path.dirname(self.csv_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._file = open(self.csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=METRICS_COLUMNS)
            self._writer.writeheader()
        self._writer.writerow({key: metrics.get(key, "") for key in METRICS_COLUMNS})
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None


def reduce_loss(
    loss_tensor: torch.Tensor,
    dp_world_size: int,
    group: dist.ProcessGroup | None = None,
) -> float:
    """Return the data-parallel mean of a scalar loss tensor."""
    loss_clone = loss_tensor.clone().detach()
    dist.all_reduce(loss_clone, op=dist.ReduceOp.SUM, group=group)
    return (loss_clone / dp_world_size).item()


def reduce_max(value: float, group: dist.ProcessGroup | None = None) -> float:
    """Return max(value) across ranks in ``group``."""
    tensor = torch.tensor([value], device=torch.cuda.current_device())
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    return tensor.item()
