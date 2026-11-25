# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Driver-side data loading: exact microbatch consumption and resumable position."""

from torch.utils.data import DataLoader

from ray_deepspeed_pipeline.errors import StepFailed, ValidationError


def build_training_dataloader(training_data, collate_fn, rows: int) -> DataLoader:
    """One loader entry is one global microbatch of `rows` rows. A short last
    entry is dropped: every microbatch of a step must have the rows the
    stages split."""
    return DataLoader(training_data, batch_size=rows, collate_fn=collate_fn, drop_last=True)


def take_microbatch_entries(data_iter, n: int) -> list[tuple]:
    """Consume exactly n (inputs, labels) entries, leaving any surplus unread.

    Raises StepFailed if the iterator runs out partway.
    """
    entries = []
    for i in range(n):
        try:
            entry = next(data_iter)
        except StopIteration:
            raise StepFailed(
                f"data iterator exhausted after {i} of {n} required "
                f"microbatch entries; the whole step fails"
            ) from None
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValidationError(
                f"entry {i} violates the v1 data contract: expected a 2-tuple "
                f"(inputs, labels), got {type(entry).__name__} "
                f"of length {len(entry) if hasattr(entry, '__len__') else '?'}"
            )
        entries.append((entry[0], entry[1]))
    return entries


class CountingIterator:
    """Iterator wrapper counting entries pulled; the checkpointed data position."""

    def __init__(self, iterator, start: int = 0):
        self._iterator = iterator
        self.consumed = start

    def __iter__(self):
        return self

    def __next__(self):
        entry = next(self._iterator)
        self.consumed += 1
        return entry


def data_position(loader: DataLoader, consumed: int) -> dict:
    """Serializable loader position. Raises ValidationError unless the loader
    reads a map-style dataset in order, the only case that resumes exactly."""
    from torch.utils.data import IterableDataset, SequentialSampler

    if isinstance(loader.dataset, IterableDataset) or not isinstance(
        loader.sampler, SequentialSampler
    ):
        raise ValidationError(
            "the engine-owned data loader cannot be resumed exactly (iterable "
            "dataset or non-sequential sampler); pass training_data=None and "
            "an explicit data_iter, and checkpoint your data position yourself "
            "via client_state"
        )
    return {"owner": "engine", "entries_consumed": int(consumed)}


def resume_loader_iter(loader: DataLoader, consumed: int) -> CountingIterator:
    """A fresh iterator starting at entry `consumed`, without replaying the
    skipped batches."""
    from torch.utils.data import Subset

    start = consumed * loader.batch_size
    rest = Subset(loader.dataset, range(min(start, len(loader.dataset)), len(loader.dataset)))
    resumed = DataLoader(
        rest, batch_size=loader.batch_size, collate_fn=loader.collate_fn, drop_last=loader.drop_last
    )
    return CountingIterator(iter(resumed), start=consumed)
