import pytest

import yaml

import pydantic

from deepspeedometer import BenchmarkRunner, parse_args_to_configs


def test_config(benchmark_args):
    benchmark_config, client_config = parse_args_to_configs(benchmark_args)


@pytest.mark.parametrize("model", [""])
def test_config_required_fail(benchmark_args):
    with pytest.raises(pydantic.ValidationError):
        benchmark_config, client_config = parse_args_to_configs(benchmark_args)


@pytest.mark.parametrize("num_config_files", [1])
def test_config_file(benchmark_args, config_files, num_clients):
    # Create a config that would generate 6 benchmark settings
    config = {"max_prompt_length": [500, 1300, 2600], "num_clients": [1, 2]}
    with open(config_files[0], "w") as f:
        yaml.dump(config, f)

    benchmark_config, client_config = parse_args_to_configs(benchmark_args)
    benchmark_runner = BenchmarkRunner(benchmark_config, client_config)
    benchmark_settings = sum(1 for _ in benchmark_runner._benchmark_settings()) * len(
        num_clients
    )
    assert benchmark_settings == 6
