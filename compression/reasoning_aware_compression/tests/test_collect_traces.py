# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""CPU-only tests for the trace-collection helpers.

Only the pieces that do not need a model are covered: the rank-coordination
helpers used by the ZeRO-Inference backend, which must degrade to no-ops when
the script runs in a single process.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collect_traces import _max_total_across_ranks, assert_ranks_agree  # noqa: E402

CPU = torch.device("cpu")


def test_max_total_across_ranks_is_a_passthrough_without_distributed():
    assert _max_total_across_ranks(0, CPU) == 0
    assert _max_total_across_ranks(1_234_567, CPU) == 1_234_567


def test_assert_ranks_agree_is_a_noop_without_distributed():
    generated = torch.randint(0, 100, (2, 8))
    assert assert_ranks_agree(generated, CPU) is None
