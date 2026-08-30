# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import argparse

import pytest

from benchmarks.opsd.benchmark_continuous_batching import (
    _build_result,
    group_request_indices,
    static_computed_tokens,
    summarize_latencies,
    validate_args,
)


def test_static_groups_respect_capacity_and_order():
    assert group_request_indices([32, 64, 96, 128, 7], 2) == [(0, 1), (2, 3), (4,)]
    assert static_computed_tokens([32, 64, 96, 128, 7], 2) == 2 * 64 + 2 * 128 + 7


def test_static_computed_tokens_does_not_use_one_large_batch():
    assert static_computed_tokens([32, 64, 96, 128], 2) == 384
    assert static_computed_tokens([32, 64, 96, 128], 2) != 4 * 128


def test_percentile_summary_matches_opsd_nearest_rank():
    assert summarize_latencies([4.0, 1.0, 3.0, 2.0]) == {"mean": 2.5, "p50": 2.0, "p95": 4.0}


def test_validate_args_rejects_non_greedy():
    args = argparse.Namespace(dtype="fp16", prompt_length=4, response_lengths=[2], max_batch_size=1,
                              warmup=0, iterations=1, temperature=0.2, seed=1)
    with pytest.raises(ValueError, match="greedy"):
        validate_args(args)


def test_validate_args_accepts_defaults_shape():
    args = argparse.Namespace(dtype="fp16", prompt_length=512, response_lengths=[32, 64], max_batch_size=2,
                              warmup=1, iterations=3, temperature=0.0, seed=1234)
    validate_args(args)


def test_result_comparisons_use_mean_latency_and_useful_throughput():
    args = argparse.Namespace(model="m", dtype="fp16", prompt_length=8, response_lengths=[2, 4],
                              max_batch_size=1, warmup=1, iterations=1, temperature=0.0, seed=1)
    modes = {
        "sequential_eager": {"latency_ms": {"mean": 10.0, "p50": 10.0, "p95": 10.0},
                             "useful_tokens": 6, "computed_tokens": 6,
                             "useful_tokens_per_second": 600.0, "peak_memory_mb": 1.0},
        "static_batch": {"latency_ms": {"mean": 8.0, "p50": 8.0, "p95": 8.0},
                          "useful_tokens": 6, "computed_tokens": 6,
                          "useful_tokens_per_second": 750.0, "peak_memory_mb": 1.0},
        "continuous_batch": {"latency_ms": {"mean": 5.0, "p50": 5.0, "p95": 5.0},
                              "useful_tokens": 6, "computed_tokens": 6,
                              "useful_tokens_per_second": 1200.0, "peak_memory_mb": 1.0},
    }
    result = _build_result(args, {"gpu": "cpu"}, modes, 6)
    assert result["comparisons"]["continuous_vs_sequential"]["latency_change_percent"] == -50.0
    assert result["comparisons"]["continuous_vs_sequential"]["useful_throughput_change_percent"] == 100.0
