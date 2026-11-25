# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Each step consumes exactly its global microbatches from the iterator."""

import pytest

from ray_deepspeed_pipeline.data import take_microbatch_entries
from ray_deepspeed_pipeline.errors import StepFailed, ValidationError


def entries(n):
    return [(f"in{i}", f"label{i}") for i in range(n)]


def test_exact_global_microbatches():
    it = iter(entries(4))
    got = take_microbatch_entries(it, 4)
    assert got == entries(4)


def test_exhausted_fails_step():
    it = iter(entries(3))
    with pytest.raises(StepFailed, match="3 of 4"):
        take_microbatch_entries(it, 4)


def test_surplus_not_consumed():
    it = iter(entries(5))
    take_microbatch_entries(it, 4)
    assert next(it) == ("in4", "label4"), "the 5th entry must remain untouched"


def test_non_2tuple_entry_rejected():
    with pytest.raises(ValidationError, match="2-tuple"):
        take_microbatch_entries(iter([("a", "b", "c")]), 1)
    with pytest.raises(ValidationError):
        take_microbatch_entries(iter(["just-inputs"]), 1)


def test_list_entries_normalized_to_tuples():
    # DataLoader batches often arrive as lists
    got = take_microbatch_entries(iter([["x", "y"]]), 1)
    assert got == [("x", "y")]


def test_training_loader_drops_a_short_last_entry():
    from ray_deepspeed_pipeline.data import build_training_dataloader

    # a short entry would change the boundary shapes mid-step
    loader = build_training_dataloader(list(range(5)), lambda rows: rows, rows=2)
    assert list(loader) == [[0, 1], [2, 3]]
