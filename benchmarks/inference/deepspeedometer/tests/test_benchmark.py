import pytest

from deepspeedometer import parse_args_to_configs, BenchmarkRunner


def test_benchmark_runner(benchmark_args, num_clients):
    benchmark_config, client_config = parse_args_to_configs(benchmark_args)
    benchmark_runner = BenchmarkRunner(benchmark_config, client_config)
    benchmark_runner.run()

    expected_results = sum(1 for _ in benchmark_runner._benchmark_settings()) * len(
        num_clients
    )
    actual_results = len(list(benchmark_runner._get_output_dir().glob("*.json")))
    assert (
        expected_results == actual_results
    ), f"Number of result files ({actual_results}) does not match expected number ({expected_results})."
