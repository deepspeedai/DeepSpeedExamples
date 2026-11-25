# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""bench/mimo/bench.sh: which runs.sh cases a layout launches, in what order,
and what it skips. DRY_RUN=1 prints the plan instead of running it."""

import os
import subprocess

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
BENCH = os.path.join(ROOT, "bench", "mimo", "bench.sh")


def plan(tmp_path, *layouts, model="Qwen3.5-4B", **env):
    out = subprocess.run(
        ["bash", BENCH, *layouts],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HOME": str(tmp_path), "DRY_RUN": "1", "MODEL": model, **env},
    )
    return out.returncode, out.stdout + out.stderr


def test_each_layout_runs_megatron_then_rdsp(tmp_path):
    code, out = plan(tmp_path, "noncoloc", "tp2", "tp4", "coloc")
    assert code == 0, out
    cases = [line.split()[-1] for line in out.splitlines() if line.startswith("run ")]
    assert cases == [
        "convert",
        "export",
        "noncoloc-mimo",
        "noncoloc-rdsp",
        "shared-megatron",
        "shared-rdsp",
        "tp4pp2-megatron",
        "tp4pp2-rdsp",
        "coloc-rdsp",
    ]


def test_mimo_means_every_non_colocated_mimo_pair(tmp_path):
    code, out = plan(tmp_path, "mimo")
    assert code == 0, out
    cases = [line.split()[-1] for line in out.splitlines() if line.startswith("run ")][2:]
    assert cases == [
        "noncoloc-mimo",
        "noncoloc-rdsp",
        "vision2-mimo",
        "vision2-rdsp",
        "langpp2-mimo",
        "langpp2-rdsp",
        "langtp4-mimo",
        "langtp4-rdsp",
    ]


def test_all_means_every_layout(tmp_path):
    assert plan(tmp_path, "all")[1] == plan(tmp_path, "noncoloc", "tp2", "tp4", "coloc")[1]


def test_layouts_without_a_recipe_for_the_model_are_skipped(tmp_path):
    code, out = plan(tmp_path, "all", model="Qwen3.5-2B")
    assert code == 0, out
    assert "skip tp4: no recipes/qwen35-2b-tp4pp2dp1.sh" in out
    assert "skip coloc: no recipes/qwen35-2b-coloc-tp2pp2dp2.sh" in out
    cases = [line.split()[-1] for line in out.splitlines() if line.startswith("run ")]
    assert "noncoloc-rdsp" in cases and "tp4pp2-megatron" not in cases


def test_existing_checkpoints_and_samples_are_reused(tmp_path):
    work = tmp_path / "mimo"
    (work / "std" / "Qwen3.5-4B").mkdir(parents=True)
    (work / "Qwen3.5-4B-mimo").mkdir()
    steps = work / "cord_steps-Qwen3.5-4B"
    steps.mkdir()
    for i in range(50):
        (steps / f"step_{i:05d}.pt").touch()
    out = plan(tmp_path, "tp2")[1]
    cases = [line.split()[-1] for line in out.splitlines() if line.startswith("run ")]
    assert cases == ["shared-megatron", "shared-rdsp"]


def test_rounds_get_their_own_log_directories(tmp_path):
    out = plan(tmp_path, "tp2", ROUNDS="2")[1]
    assert f"{tmp_path}/bench/Qwen3.5-4B/r1" in out and f"{tmp_path}/bench/Qwen3.5-4B/r2" in out


@pytest.mark.parametrize("layouts", [(), ("tp8",)])
def test_unknown_or_missing_layout_prints_usage(tmp_path, layouts):
    code, out = plan(tmp_path, *layouts)
    assert code != 0 and "noncoloc" in out and "coloc" in out
