# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""bench/mimo/nsys_breakdown.py: per-GPU time per training step from an Nsight
Systems trace (exported to SQLite), the same table for Megatron and rdsp."""

import os
import sqlite3
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "bench", "mimo"))

import nsys_breakdown  # noqa: E402

MS = 1_000_000  # nsys timestamps are nanoseconds


def trace(path, kernels):
    """A minimal nsys SQLite export: kernels as (device, start ms, end ms, name)."""
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    db.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
        "(start INTEGER, end INTEGER, deviceId INTEGER, shortName INTEGER)"
    )
    names = {}
    for device, start, end, name in kernels:
        if name not in names:
            names[name] = len(names) + 1
            db.execute("INSERT INTO StringIds VALUES (?, ?)", (names[name], name))
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)",
            (start * MS, end * MS, device, names[name]),
        )
    db.commit()
    db.close()


@pytest.fixture
def two_gpus(tmp_path):
    """200 ms window, 2 steps of 100 ms. GPU 0 (language: runs the linear
    attention kernel): 40 ms matmul, 20 ms linear attention, 30 ms TP
    all-reduce overlapping 10 ms of the matmul. GPU 1 (vision): 50 ms matmul."""
    path = str(tmp_path / "run.sqlite")
    trace(
        path,
        [
            (0, 0, 40, "sm90_xmma_gemm_bf16bf16_bf16f32"),
            (0, 30, 60, "ncclDevKernel_AllReduce_Sum_bf16_RING_LL"),
            (0, 60, 80, "chunk_gated_delta_rule_fwd_kernel_h"),
            (0, 199, 200, "vectorized_elementwise_kernel"),
            (1, 0, 50, "nvjet_tst_128x256_64x4_1x2_h_bz_coopB_NNT"),
            (1, 150, 150, "vectorized_elementwise_kernel"),
        ],
    )
    return path


def test_kernels_fall_into_matmul_attention_communication_and_other():
    assert nsys_breakdown.category("sm90_xmma_gemm_bf16bf16_bf16f32") == "matmul"
    assert nsys_breakdown.category("nvjet_tst_128x256_64x4_1x2_h_bz_coopB_NNT") == "matmul"
    assert nsys_breakdown.category("flash_fwd_kernel") == "attention"
    assert nsys_breakdown.category("chunk_gated_delta_rule_fwd_kernel_h") == "linear attention"
    assert nsys_breakdown.category("ncclDevKernel_AllReduce_Sum_bf16_RING_LL") == "all-reduce"
    assert nsys_breakdown.category("ncclDevKernel_SendRecv") == "send/recv"
    assert nsys_breakdown.category("vectorized_elementwise_kernel") == "other"


def test_per_gpu_busy_idle_and_categories_per_step(two_gpus):
    rows = {r["device"]: r for r in nsys_breakdown.breakdown(two_gpus, step_ms=100)}
    lang, vision = rows[0], rows[1]
    assert lang["role"] == "language" and vision["role"] == "vision"
    assert lang["steps"] == pytest.approx(2.0)
    # busy is the union of kernel intervals (0-80 and 199-200), per step
    assert lang["busy_ms"] == pytest.approx(40.5)
    assert lang["idle_ms"] == pytest.approx(59.5)
    assert lang["all-reduce"] == pytest.approx(15.0)
    assert lang["matmul"] == pytest.approx(20.0)
    assert lang["kernels"] == pytest.approx(2.0)
    assert vision["busy_ms"] == pytest.approx(25.0)


def test_main_prints_one_row_per_gpu(two_gpus, capsys):
    nsys_breakdown.main([two_gpus, "--step-ms", "100"])
    out = capsys.readouterr().out.splitlines()
    assert out[0].split()[:3] == ["gpu", "role", "busy"]
    assert [line.split()[:2] for line in out[1:3]] == [["0", "language"], ["1", "vision"]]
