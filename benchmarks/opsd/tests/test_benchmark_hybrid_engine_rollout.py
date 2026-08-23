# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import importlib.util
import json
from pathlib import Path
import unittest


_BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "benchmark_hybrid_engine_rollout.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_hybrid_engine_rollout", _BENCHMARK_PATH)
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


class BenchmarkHybridEngineRolloutTest(unittest.TestCase):

    def test_argument_parsing(self):
        args = benchmark._build_parser().parse_args([
            "--model",
            "local-model",
            "--dtype",
            "bf16",
            "--batch-sizes",
            "2",
            "4",
            "--samples-per-prompt",
            "1",
            "3",
            "--prompt-lengths",
            "8",
            "--response-lengths",
            "16",
            "--release-inference-cache",
        ])

        self.assertEqual(args.model, "local-model")
        self.assertEqual(args.dtype, "bf16")
        self.assertEqual(args.batch_sizes, [2, 4])
        self.assertEqual(args.samples_per_prompt, [1, 3])
        self.assertEqual(args.prompt_lengths, [8])
        self.assertEqual(args.response_lengths, [16])
        self.assertTrue(args.release_inference_cache)

    def test_summary_percentiles(self):
        profiles = []
        for value in range(1, 21):
            profiles.append({field: float(value) for field in benchmark._TIMING_FIELDS})

        summary = benchmark._summarize(profiles)

        self.assertEqual(summary["total_ms"], {"mean": 10.5, "p50": 10.5, "p95": 19.0})
        self.assertEqual(summary["tokens_per_second"]["p95"], 19.0)

    def test_largest_effective_batch_executes_first_without_reordering_results(self):
        args = benchmark._build_parser().parse_args([
            "--batch-sizes",
            "1",
            "2",
            "--samples-per-prompt",
            "1",
            "4",
            "--prompt-lengths",
            "8",
            "--response-lengths",
            "8",
        ])

        requested, execution_order = benchmark._ordered_case_specs(args)

        self.assertEqual(requested, [(1, 1, 8, 8), (1, 4, 8, 8), (2, 1, 8, 8), (2, 4, 8, 8)])
        self.assertEqual([requested[index] for index in execution_order], [
            (2, 4, 8, 8),
            (1, 4, 8, 8),
            (2, 1, 8, 8),
            (1, 1, 8, 8),
        ])

    def test_json_output_has_basic_fields(self):
        args = benchmark._build_parser().parse_args(["--model", "local-model", "--iterations", "1"])
        case = {
            "batch_size": 1,
            "samples_per_prompt": 1,
            "prompt_length": 8,
            "requested_response_length": 8,
            "returned_response_length": 8,
            "peak_memory_mb": 12.5,
            "summary": {},
            "profiles": [{"total_ms": 1.0}],
        }

        result = json.loads(json.dumps(benchmark._build_result(args, "cuda:0", [case])))

        self.assertEqual(set(result), {
            "model",
            "dtype",
            "device",
            "warmup",
            "iterations",
            "temperature",
            "top_p",
            "release_inference_cache",
            "cases",
        })
        self.assertEqual(result["cases"][0]["peak_memory_mb"], 12.5)
        self.assertEqual(result["cases"][0]["profiles"], [{"total_ms": 1.0}])


if __name__ == "__main__":
    unittest.main()
