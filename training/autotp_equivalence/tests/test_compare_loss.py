# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Tests for the AutoTP loss-curve comparison."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPARE = ROOT / "compare_loss.py"


def _write(path: Path, losses, autotp_size=1) -> None:
    with open(path, "w") as f:
        for step, loss in enumerate(losses):
            f.write(json.dumps({"step": step, "loss": loss, "autotp_size": autotp_size}) + "\n")


def _compare(baseline: Path, candidate: Path, *extra):
    return subprocess.run([sys.executable, str(COMPARE), str(baseline), str(candidate), *extra],
                          capture_output=True,
                          text=True)


def test_identical_curves_agree(tmp_path):
    baseline, candidate = tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    _write(baseline, [2.0, 1.5, 1.25])
    _write(candidate, [2.0, 1.5, 1.25], autotp_size=3)

    result = _compare(baseline, candidate)
    assert result.returncode == 0, result.stderr
    assert "worst_rel=0.00e+00" in result.stdout


def test_reassociation_noise_is_accepted(tmp_path):
    baseline, candidate = tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    _write(baseline, [2.0, 1.5, 1.25])
    _write(candidate, [2.0 + 1e-6, 1.5 - 1e-6, 1.25 + 1e-6], autotp_size=3)

    result = _compare(baseline, candidate)
    assert result.returncode == 0, result.stderr
    assert "OK: 3 steps agree" in result.stdout


def test_a_drifting_curve_is_rejected(tmp_path):
    baseline, candidate = tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    _write(baseline, [2.0, 1.5, 1.25])
    # A wrongly sharded run trains a different model, so the gap compounds.
    _write(candidate, [2.0, 1.6, 1.6], autotp_size=3)

    result = _compare(baseline, candidate)
    assert result.returncode == 1
    assert "diverges" in result.stderr


def test_the_worst_step_is_reported_not_the_last(tmp_path):
    baseline, candidate = tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    _write(baseline, [1.0, 1.0, 1.0])
    _write(candidate, [1.0, 1.5, 1.0], autotp_size=3)

    result = _compare(baseline, candidate)
    assert result.returncode == 1
    assert "at step 1" in result.stderr


def test_the_worst_step_is_always_printed(tmp_path):
    """A long run prints a sample of steps, but never hides the worst one."""
    baseline, candidate = tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    losses = [1.0] * 200
    _write(baseline, losses)
    spiked = list(losses)
    spiked[137] = 1.0001
    _write(candidate, spiked, autotp_size=3)

    result = _compare(baseline, candidate, "--print-every", "50")
    assert result.returncode == 0, result.stderr
    assert "<- worst" in result.stdout
    worst_rows = [line for line in result.stdout.splitlines() if "<- worst" in line]
    assert len(worst_rows) == 1
    assert worst_rows[0].split()[0] == "137"


def test_runs_with_no_shared_steps_fail(tmp_path):
    baseline, candidate = tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    _write(baseline, [1.0])
    candidate.write_text(json.dumps({"step": 7, "loss": 1.0}) + "\n")

    result = _compare(baseline, candidate)
    assert result.returncode == 1
    assert "no steps in common" in result.stderr


def test_a_truncated_run_warns_but_still_compares(tmp_path):
    baseline, candidate = tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    _write(baseline, [1.0, 1.0, 1.0])
    _write(candidate, [1.0, 1.0], autotp_size=3)

    result = _compare(baseline, candidate)
    assert result.returncode == 0, result.stderr
    assert "warning: comparing 2 shared steps" in result.stderr


def test_a_wrong_forward_is_caught_at_the_first_step(tmp_path):
    """A gap before any weight update cannot be blamed on training dynamics."""
    baseline, candidate = tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    _write(baseline, [2.0, 1.5, 1.25])
    _write(candidate, [2.001, 1.5, 1.25], autotp_size=3)

    result = _compare(baseline, candidate)
    assert result.returncode == 1
    assert "the sharded forward is wrong" in result.stderr
    # It must not be excused as drift: the overall gap is well inside --rtol.
    assert "diverges" not in result.stderr


def test_drift_after_the_first_step_is_not_a_forward_error(tmp_path):
    """Later steps are held to the looser tolerance, since dynamics amplify noise."""
    baseline, candidate = tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    _write(baseline, [2.0, 1.5, 1.25])
    _write(candidate, [2.0, 1.5008, 1.2506], autotp_size=3)

    result = _compare(baseline, candidate)
    assert result.returncode == 0, result.stderr
    assert "first_rel=0.00e+00" in result.stdout


def test_the_forward_tolerance_is_configurable(tmp_path):
    baseline, candidate = tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    _write(baseline, [2.0, 1.5])
    _write(candidate, [2.001, 1.5], autotp_size=3)

    assert _compare(baseline, candidate, "--forward-rtol", "1e-2").returncode == 0


@pytest.mark.parametrize("rtol, expected", [("1e-1", 0), ("1e-4", 1)])
def test_the_tolerance_flag_is_honored(tmp_path, rtol, expected):
    baseline, candidate = tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    # Step 0 matches, so only --rtol decides the outcome.
    _write(baseline, [1.0, 1.0])
    _write(candidate, [1.0, 1.01], autotp_size=3)

    assert _compare(baseline, candidate, "--rtol", rtol).returncode == expected
