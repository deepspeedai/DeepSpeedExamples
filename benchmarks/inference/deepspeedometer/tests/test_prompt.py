import pytest

from deepspeedometer import BenchmarkRunner, parse_args_to_configs


@pytest.mark.parametrize("prompt_length_var, max_new_tokens_var", [(0, 0)])
def test_prompt_length(benchmark_args):
    benchmark_config, client_config = parse_args_to_configs(benchmark_args)
    benchmark_runner = BenchmarkRunner(benchmark_config, client_config)
    num_clients, prompt_config = next(benchmark_runner._benchmark_settings())

    for prompt in benchmark_runner.prompt_generator(prompt_config, num_prompts=10):
        prompt_length = benchmark_runner.prompt_generator.count_tokens(prompt.text)
        # Using pytest.approx here because often we will have 1-off errors due to tokenization special tokens
        assert prompt_length == pytest.approx(benchmark_runner.config.prompt_length, 1)
