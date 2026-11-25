# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""bench/common/summarize.py pairs each Megatron run with its rdsp run and
prints one results row per pair."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "bench", "common"))

import pytest  # noqa: E402
import summarize  # noqa: E402

PUBLISHED = os.path.join(HERE, "..", "..", "bench", "qwen35_small", "logs")
# the 2B/4B/H100 logs are not in the repository; the tests that read them run where they are
needs_published_logs = pytest.mark.skipif(
    not os.path.isdir(PUBLISHED), reason="bench/qwen35_small/logs not here"
)


def _megatron_log(ms: float) -> str:
    return "".join(
        f" iteration {i + 1:8d}/      50 | consumed samples: {64 * (i + 1)} | "
        f"elapsed time per iteration (ms): {ms:.1f} | lm loss: 5.0E-01 |\n"
        for i in range(50)
    )


def _rdsp_log(ms: int) -> str:
    return "".join(
        f"step {i} loss 0.5000 {ms} ms 1 real tok/s 2 padded tok/s real 65000 supervised 13000\n"
        for i in range(50)
    )


def test_pairs_megatron_and_rdsp_logs(tmp_path, capsys):
    (tmp_path / "shared-megatron.log").write_text(_megatron_log(10000.0))
    (tmp_path / "shared-rdsp.log").write_text(_rdsp_log(8000))
    summarize.main([str(tmp_path)])
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if "TP2xPP2xDP2" in line)
    assert "10.00" in row and "8.00" in row and "-20%" in row
    per_gpu = row.split()[-2:]  # tokens/s per GPU: 64 rows x 2048 over 10 s / 8 s, 8 GPUs
    assert per_gpu == ["1638", "2048"]
    assert "TP4xPP2xDP1" not in out  # pairs without logs are left out


def test_tokens_per_second_follow_rows_and_sequence_length(tmp_path, capsys):
    (tmp_path / "shared-megatron.log").write_text(_megatron_log(10000.0))
    (tmp_path / "shared-rdsp.log").write_text(_rdsp_log(8000))
    summarize.main([str(tmp_path), "--rows", "128", "--seq", "1024"])
    row = next(line for line in capsys.readouterr().out.splitlines() if "TP2xPP2xDP2" in line)
    assert row.split()[-5:-3] == ["13107", "16384"]  # 128 x 1024 over 10 s and 8 s


def test_mimo_layout_gets_its_row_on_five_gpus(tmp_path, capsys):
    (tmp_path / "noncoloc-mimo.log").write_text(_megatron_log(10000.0))
    (tmp_path / "noncoloc-rdsp.log").write_text(_rdsp_log(8000))
    summarize.main([str(tmp_path)])
    row = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("vision 1 GPU + language TP2xDP2")
    )
    assert "-20%" in row and row.split()[-2:] == ["2621", "3277"]  # 64 x 2048 / 10 s, 8 s, 5 GPUs


def test_rdsp_only_layout_gets_a_row_without_megatron(tmp_path, capsys):
    """Colocated vision has no Megatron counterpart: its row shows rdsp alone."""
    (tmp_path / "coloc-rdsp.log").write_text(_rdsp_log(5000))
    summarize.main([str(tmp_path)])
    row = next(line for line in capsys.readouterr().out.splitlines() if "colocated" in line)
    assert "5.00" in row and row.count("-") >= 2


@needs_published_logs
def test_published_2b_and_4b_logs_give_the_benchmark_results_numbers(capsys):
    """bench/qwen35_small/logs holds the logs behind BENCHMARK_RESULTS.md's PCIe table;
    summarize.py on them prints every step time it reports."""
    rows = []
    for size in ("2b", "4b"):
        summarize.main([os.path.join(PUBLISHED, size)])
        rows += capsys.readouterr().out.splitlines()[1:]
    got = [tuple(line.split()[-8:-6]) for line in rows]
    assert got == [("9.64", "8.30"), ("7.39", "5.30"), ("10.04", "8.69"), ("15.93", "13.29")]
    with open(os.path.join(HERE, "..", "..", "docs", "BENCHMARK_RESULTS.md")) as f:
        doc = f.read()
    for megatron, rdsp in got:
        assert f"| {megatron} s | {rdsp} s |" in doc


@needs_published_logs
def test_published_h100_logs_give_the_benchmark_results_numbers(capsys):
    """bench/qwen35_small/logs/h100 holds the 8x H100 logs; summarize.py on them prints
    every step time BENCHMARK_RESULTS.md reports for H100."""
    summarize.main([os.path.join(PUBLISHED, "h100")])
    rows = capsys.readouterr().out.splitlines()[1:]
    got = [tuple(line.split()[-8:-6]) for line in rows]
    assert got == [("7.99", "7.89"), ("8.79", "6.49"), ("18.24", "12.30"), ("-", "5.62")]
    with open(os.path.join(HERE, "..", "..", "docs", "BENCHMARK_RESULTS.md")) as f:
        doc = f.read()
    for megatron, rdsp in got:
        cell = "—" if megatron == "-" else f"{megatron} s"
        assert f"| {cell} | {rdsp} s |" in doc, (megatron, rdsp)


@needs_published_logs
def test_docs_report_tokens_per_gpu_from_the_logs(capsys):
    """Every published row's tokens/s per GPU (both systems) appears in
    BENCHMARK_RESULTS.md, as printed by summarize.py."""
    values = set()
    for size in ("2b", "4b", "h100"):
        summarize.main([os.path.join(PUBLISHED, size)])
        for line in capsys.readouterr().out.splitlines()[1:]:
            values.update(v for v in line.split()[-2:] if v != "-")
    with open(os.path.join(HERE, "..", "..", "docs", "BENCHMARK_RESULTS.md")) as f:
        doc = f.read()
    for v in sorted(values):
        assert f"| {int(v):,} |" in doc, v
