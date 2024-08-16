# DeepSpeedometer

NOTE: This is an experimental tool and is not currently being supported since it's not fully functional. Please use the MII benchmark which can be found here:
https://github.com/microsoft/DeepSpeedExamples/tree/master/benchmarks/inference/mii

This benchmark is designed to measure performance of LLM serving solutions. Using a different number of parallel clients sending requests to an inference server, we gather data to plot throughput-latency curves and find the saturation point of an inference server that demonstrates the maximum performance.

## Installation

To install the benchmark, clone this repository and install using `pip`:
```shell
git clone https://github.com/Microsoft/DeepSpeedExamples
cd ./DeepSpeedExamples/benchmarks/deepspeedometer
pip install .
```

## Usage

To quickly test the benchmark code without creating an inference server, run the following:
```
python3 -m deepspeedometer.benchmark_runner --model facebook/opt-125m --api dummy
```

### Supports APIs

The benchmark supports different APIs, each with their own client type. Depending on the client, you may need to run the benchmark against a locally hosted inference server or a remote inference server. Adding support for new serving solutions can be achieved by creating a new client class that defines a few basic methods. See the section below on adding new clients for more information.

The clients (i.e., APIs) curently supported (and configuration options for each) are listed below. You can see more information about the configuration options by looking at the `*ClientConfig` classes located in `clients/*.py`:

1. `fastgen`: Runs a local model inference server with DeepSpeed's FastGen. Config options include:
    - `model`: Which model to use for serving (required)
    - `deployment_name`: Name of the deployment server
    - `tp_size`: Tensor parallel size for each model replicas
    - `num_replicas`: Number of model replicas
    - `max_ragged_batch_size`: Max number of requests running per model replicas
    - `quantization_mode`: Type of quantization to use
2. `vllm`: Runs a local model inference server with vLLM.
    - `model`: Which model to use for serving (required)
    - `tp_size`: Tensor parallel size for model
    - `port`: Which port to use for REST API
3. `azureml`: Interfaces with remote AzureML online endpoint/deployment.
    - `api_url`: AzureML endpoint API URL (required)
    - `api_key`: AzureML token key for connecting to endpoint (required)
    - `deployment_name`: Name of deployment hosted in given endpoint (required)

### Benchmark Configuration

The Benchmark has many options for tailoring performance measurements to a specific use-cases. For additional information and default values, see the `BenchmarkConfig` class defined in `benchmark_runner.py`.

- `api`: Which API to use
- `warmup_requests`: Number of warm-up requests to run before measuring performance
- `result_dir`: Directory where results will be written out (as JSON files)
- `use_threading`: Whether to use threading for the benchmark clients. Default is to use multi-processing
- `config_file`: One or more config YAML files that contain values for any of the Prompt configuration options (see below section on prompt configuration)
- `num_clients`: One or more integer values for the number of parallel clients to run
- `num_requests_per_client`: Number of requests that will be run by each of the parallel clients
- `min_requests`: Minimum number of requests to be sent during duration of benchmark. Useful when there is a low number of clients to ensure good measurement.
- `prompt_text_source`: Text file or string that will be sampled to generate request prompts
- `early_stop_latency`: When running multiple values for `num_clients`, if the average latency per request exceeds this value (in seconds) the benchmark will not test a larger number of parallel clients
- `force`: Force the overwrite of result files. By default, if a result file exists, the benchmark is skipped

### Prompt Configuration

These options allow users to modify the prompt input and generation behavior of the served models. Note that you can run multiple prompt configurations in a single command by using the `config_file` input as described in the Benchmark Configuration section.

- `model`: Which model to use for tokenizing prompts (required)
- `prompt_generator_seed`: Seed value for random number generation
- `max_prompt_length`: The maximum prompt length allowed
- `prompt_length`: Target mean prompt length
- `prompt_lenght_var`: Variance of generated prompt lengths
- `max_new_tokens`: Target mean number of generated tokens
- `max_new_tokens_var`: Variance of generated tokens
- `streaming`: Whether to enabled streaming output for generated tokens

#### About Prompt Generation

To mimic real-world serving scenarios, this benchmark samples prompt length and generated token length values from a normal distribution. This distribution can be manipulated with the `prompt_length*` and `max_new_tokens*` values in the prompt configuration. To get all prompt lengths and generation lengths to match exactly, set the `*_var` values to 0.

## Adding New Client APIs

The DeepSpeedometer benchmark was designed to allow easily adding support for new inference server solutions. To do so:

1. Create a new `*_client.py` file in the `clients/` directory.
2. Define a `*Client` class that inherits from the `BaseClient` class in `clients/base.py`. This class should define 5 methods: `start_service`, `stop_service`, `prepare_request`, `send_request`, and `process_response`. Take a look at the type hints for these methods in the `BaseClient` class to understand the expected inputs and outputs for each method.
3. Define a `*ClientConfig` class that inherits from the `BaseConfigModel` class. Place any configuration options (i.e., user-passed command line arguments) necessary for your defined `*Client` class in here.
4. Import the newly added `*Client` and `*ClientConfig` into `clients/__init__.py` and add them to the `client_config_classes` and `client_classes` dictionaries.

For the simplest example of adding a new client, take a look at the `clients/dummy_client.py` file where we have defined a client that does not stand up a server and only returns a sample of the input prompt after a short sleep cycle. We use this as a light-weight class for unit testing.
