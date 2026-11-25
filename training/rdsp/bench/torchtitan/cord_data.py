# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Megatron-Bridge's exported CORD-v2 rows as a TorchTitan dataset (see cord_qwen35.py).

Source, processor and collator only: no model or kernel code, so the data path runs
and is tested on a CPU.
"""

import glob
import os
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torchtitan.components.data import GrainDataLoader, SampleProcessor, SingleDatasetConfig
from torchtitan.components.data.multimodal.mm_collator import MultiModalCollator
from torchtitan.components.data.types import DatasetBuildContext
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config import Configurable

ROWS_PER_STEP = int(os.environ.get("ROWS", "64"))  # rows of 2048 per step: 64, 128 on two nodes


class ExportedStepsSource(Configurable):
    """Rows of an export directory in training order: step file by step file, row
    by row. A step file (one training step's rows) is loaded once and kept while
    its rows are read."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        directory: str
        rows_per_step: int = ROWS_PER_STEP
        data_parallel: int = 1
        """TorchTitan gives data-parallel rank r the r-th contiguous slice of the
        dataset; Megatron gives it row r of each group of `data_parallel` consecutive
        rows. With data_parallel = d, index i serves row (i % n) * d + i // n (n = rows
        / d), so rank r's slice is rows r, r + d, r + 2d, ... and each microbatch holds
        the same d consecutive rows as Megatron's."""

    def __init__(self, config: Config) -> None:
        self.paths = sorted(glob.glob(os.path.join(config.directory, "step_*.pt")))
        if not self.paths:
            raise FileNotFoundError(f"no step_*.pt files in {config.directory}")
        self.rows = config.rows_per_step
        self.dp = config.data_parallel
        if (len(self.paths) * self.rows) % self.dp:
            raise ValueError(
                f"{len(self.paths) * self.rows} rows do not split over {self.dp} ranks"
            )
        self._lock = threading.Lock()
        self._cached: tuple[int, list] | None = None

    def __len__(self) -> int:
        return len(self.paths) * self.rows

    def _step(self, step: int) -> list:
        with self._lock:
            if self._cached is None or self._cached[0] != step:
                samples = torch.load(self.paths[step], weights_only=False)
                if len(samples) != self.rows:
                    raise ValueError(
                        f"{self.paths[step]}: {len(samples)} rows, expected {self.rows}"
                    )
                self._cached = (step, samples)
            return self._cached[1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard, offset = divmod(index, len(self) // self.dp)
        step, row = divmod(offset * self.dp + shard, self.rows)
        return self._step(step)[row]


def exported_row_to_sample(row: dict[str, Any]) -> dict[str, Any]:
    """One exported row (batch of 1, Hugging Face conventions) as a TorchTitan
    multimodal sample: input = all exported tokens, label t = target of input t,
    and one (patches, grid) pair per image in the row."""
    ids = row["input_ids"][0]
    hf_labels = row["labels"][0]
    labels = torch.full_like(ids, IGNORE_INDEX)
    labels[:-1] = hf_labels[1:]
    grids = row["image_grid_thw"]
    patches = torch.split(row["pixel_values"], grids.prod(-1).tolist())
    return {
        "input_ids": ids,
        "labels": labels,
        "positions": torch.arange(ids.shape[0]),
        "pixel_values": list(zip(patches, grids)),
    }


class ExportedRowProcessor(SampleProcessor):
    """exported_row_to_sample as a dataset processor (no randomness)."""

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        del config, context

    def __call__(self, sample: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
        del rng
        return exported_row_to_sample(sample)


class ExportedRowCollator(MultiModalCollator):
    """MultiModalCollator whose images arrive already patched, as (patches, grid)
    pairs: they are concatenated in order instead of patchified."""

    @dataclass(kw_only=True, slots=True)
    class Config(MultiModalCollator.Config):
        pass

    def collate_images(self, all_images: list) -> tuple[torch.Tensor, torch.Tensor]:
        patches = torch.cat([p for p, _ in all_images])
        grids = torch.stack([g for _, g in all_images])
        return patches, grids


def exported_dataloader(directory: str, data_parallel: int = 1) -> GrainDataLoader.Config:
    """Bridge's rows in Bridge's order, no shuffle. One pass without data parallelism;
    with it TorchTitan requires repeat (the trainer's step count ends the run, and a
    run of all exported steps never wraps)."""
    return GrainDataLoader.Config(
        dataset=SingleDatasetConfig(
            source=ExportedStepsSource.Config(directory=directory, data_parallel=data_parallel),
            processor=ExportedRowProcessor.Config(),
        ),
        collator=ExportedRowCollator.Config(build_mrope_positions=True),
        shuffle=False,
        repeat=data_parallel > 1,
    )
