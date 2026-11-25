# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""The README's headline step times are the medians of the logs docs/REPRODUCE.md cites,
as bench/gap_round/scripts/medians.py computes them, and its ratios follow from them.
The logs are not in the repository; these tests run where they are."""

import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "bench", "gap_round", "scripts"))

import medians  # noqa: E402

# model: (rdsp, Megatron-Bridge, TorchTitan), each a (log format, log under bench/)
LOGS = {
    "35B": (
        ("rdsp", "two_node_35b/vision_stage/logs/rdsp-k4-rc.log"),
        ("bridge", "gap_round/logs/bridge-pp2ep8-nv.log"),
        (
            "titan",
            "two_node_35b/fsdp16/logs/run-qwen35_35b_a3b_cord_fsdp16_ep8_bf16-node0.log",
        ),
    ),
    "27B": (
        ("rdsp", "bridge_vs_rdsp/27b_tp1pp8dp1/logs/rdsp-nov21fix-rows1-best.log"),
        ("bridge", "bridge_vs_rdsp/27b_tp1pp8dp1/logs/bridge-nov21fix-mbs1-best.log"),
        ("titan", "torchtitan/logs/run-qwen35_27b_cord_fsdp8_rc_bf16.log"),
    ),
}


def _logs_present(model: str) -> bool:
    return all(os.path.exists(os.path.join(ROOT, "bench", log)) for _, log in LOGS[model])


needs_logs = pytest.mark.parametrize(
    "model",
    [
        pytest.param(m, marks=pytest.mark.skipif(not _logs_present(m), reason="logs not here"))
        for m in sorted(LOGS)
    ],
)


def readme() -> str:
    with open(os.path.join(ROOT, "README.md")) as f:
        return f.read()


@needs_logs
def test_the_results_row_is_the_cited_logs_medians(model):
    ours, bridge, titan = (
        medians.median(kind, os.path.join(ROOT, "bench", log)) for kind, log in LOGS[model]
    )
    row = f"| **{ours:.2f} s** | {bridge:.2f} s | {titan:.2f} s |"
    assert row in readme()


@needs_logs
def test_the_ratios_follow_from_the_step_times(model):
    ours, bridge, titan = (
        medians.median(kind, os.path.join(ROOT, "bench", log)) for kind, log in LOGS[model]
    )
    text = readme()
    assert f"{bridge / ours:.2f}×" in text
    assert f"{titan / ours:.2f}×" in text


def test_rdsp_steps_are_counted_as_logged():
    """rdsp's log counts from step 0, and medians.py keeps its numbers."""
    log = "".join(f"step {i} loss 0.5 {1000 * (i + 1)} ms\n" for i in range(50))
    assert min(medians.rdsp(log)) == 0
    assert medians.rdsp(log)[6] == 7.0
