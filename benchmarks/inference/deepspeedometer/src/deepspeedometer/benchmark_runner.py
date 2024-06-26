import itertools
import json
import multiprocessing
import os
import queue
import sys
import threading
import time
import yaml
from pathlib import Path
from typing import List, Iterable, Tuple

from loguru import logger
from tqdm import tqdm

from .clients import client_classes, BaseClient
from .config import BaseConfigModel
from .prompt import Prompt, PromptConfig, PromptGenerator
from .response import Response
from .sample_input import sample_input_text


class BenchmarkConfig(PromptConfig):
    api: str = "azure_ml"
    """ Which API to use for benchmarking. New APIs can be added by creating a new client class in the `clients` directory. """

    warmup_requests: int = 1
    """ Number of requests to run (per client) as a warm-up before starting the benchmark. """

    result_dir: Path = Path("./results")
    """ Top directory where results will be saved. """

    use_threading: bool = False
    """ Whether to use threading or multiprocessing for parallel client requests. Default is multiprocessing. """

    config_file: List[Path] = []
    """ Path to YAML file(s) containing benchmark configuration settings. """

    num_clients: List[int] = [1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32]
    """ Number of clients to run in parallel. """

    num_requests_per_client: int = 16
    """ Number of requests to run per client. """

    min_requests: int = 128
    """ Minimum number of request to create (regardless of num_requests_per_client). """

    prompt_text_source: str = sample_input_text
    """ Text file or string to use for generated prompts. """

    early_stop_latency: float = 10.0
    """ Maximum mean latency (in seconds) to allow before stopping the benchmark early. """

    force: bool = False
    """ Whether to overwrite existing result files. """


class ClientLauncher:
    def __init__(
        self,
        client_class: BaseClient,
        client_config: BaseConfigModel,
        warmup_requests: int,
        use_threading: bool,
        prompt_generator: PromptGenerator,
    ):
        self.client_class = client_class
        self.client_config = client_config
        self.client_obj = client_class(client_config)
        self.warmup_requests = warmup_requests
        self.prompt_generator = prompt_generator

        if use_threading:
            self.runnable_cls = threading.Thread
            self.barrier_cls = threading.Barrier
            self.queue_cls = queue.Queue
        else:
            self.runnable_cls = multiprocessing.Process
            self.barrier_cls = multiprocessing.Barrier
            self.queue_cls = multiprocessing.Queue

    def run_parallel_clients(self, num_clients: int) -> None:
        logger.info(f"Launching {num_clients} client(s)")

        total_requests = self.request_queue.qsize()

        self.barrier = self.barrier_cls(num_clients + 1)
        processes = [
            self.runnable_cls(
                target=self._run_client,
                args=(
                    i,
                    self.barrier,
                    self.request_queue,
                    self.response_queue,
                    self.client_class,
                    self.client_config,
                    self.warmup_requests,
                ),
            )
            for i in range(num_clients)
        ]
        for p in processes:
            p.start()

        self.barrier.wait()  # Barrier 1 for master process

        self._progress_bar(total_requests - self.warmup_requests * num_clients)

        self.barrier.wait()  # Barrier 2 for master process

    def _progress_bar(self, total_requests: int) -> None:
        pbar = tqdm(total=total_requests)
        num_responses = 0
        while num_responses != total_requests:
            num_responses = self.response_queue.qsize()
            pbar.update(num_responses - pbar.n)
            time.sleep(1)
        pbar.close()

    @staticmethod
    def _run_client(
        client_id: int,
        barrier: multiprocessing.Barrier,
        request_queue: multiprocessing.Queue,
        response_queue: multiprocessing.Queue,
        client_class: BaseClient,
        client_config: BaseConfigModel,
        warmup_requests: int,
    ):
        client = client_class(client_config)

        for _ in range(warmup_requests):
            prompt = request_queue.get(timeout=1.0)
            _ = client.send_request(prompt.request_kwargs)

        barrier.wait()  # Barrier 1 for client process
        try:
            while True:
                prompt = request_queue.get(timeout=1.0)
                start_time = time.time()
                raw_response = client.send_request(prompt.request_kwargs)
                end_time = time.time()
                request_time = end_time - start_time
                response = Response(
                    prompt_text=prompt.text,
                    prompt_tokens=prompt.num_tokens,
                    raw_response=raw_response,
                    request_time=request_time,
                    client_id=client_id,
                )
                response_queue.put_nowait(response)
        except queue.Empty:
            pass

        barrier.wait()  # Barrier 2 for client process

    def add_request(self, prompt: Prompt) -> None:
        request_kwargs = self.client_obj.prepare_request(prompt)
        prompt.request_kwargs = request_kwargs
        self.request_queue.put(prompt)

    def get_response(self) -> Response:
        response = self.response_queue.get(timeout=1.0)
        processed_response = self.client_obj.process_response(response.raw_response)
        response.generated_output = processed_response
        response.generated_tokens = self.prompt_generator.count_tokens(
            processed_response
        )
        return response

    def clear_queues(self) -> None:
        self.request_queue = self.queue_cls()
        self.response_queue = self.queue_cls()

    def start_service(self) -> None:
        self.client_obj.start_service()

    def stop_service(self) -> None:
        self.client_obj.stop_service()


class BenchmarkRunner:
    def __init__(
        self, benchmark_config: BaseConfigModel, client_config: BaseConfigModel
    ) -> None:
        logger.info("Initializing Benchmark Runner")
        self.config = benchmark_config
        self.client_config = client_config
        self.client_class = client_classes[self.config.api]
        self.prompt_generator = PromptGenerator(
            self.config.model, self.config.prompt_text_source
        )
        self.client_launcher = ClientLauncher(
            client_class=self.client_class,
            client_config=self.client_config,
            warmup_requests=self.config.warmup_requests,
            use_threading=self.config.use_threading,
            prompt_generator=self.prompt_generator,
        )
        self.all_responses = []

    def _benchmark_settings(self) -> Iterable[Tuple[List[int], PromptConfig]]:
        prompt_config_keys = list(PromptConfig.model_fields.keys())

        configs_list = []
        for f in self.config.config_file:
            logger.info(f"Generating benchmark run settings from config file: {f}")
            with open(f, "r") as fh:
                file_config = yaml.safe_load(fh)

            # Get any prompt config values stored in config files
            for key in prompt_config_keys + ["num_clients"]:
                if key not in file_config:
                    file_config[key] = getattr(self.config, key)
            configs_list.append(file_config)

        if not configs_list:
            logger.info(f"Generating benchmark run settings from command line args")
            configs_list.append(
                {
                    key: getattr(self.config, key)
                    for key in prompt_config_keys + ["num_clients"]
                }
            )

        all_config_product = []
        for config in configs_list:
            # Ensure all config values are iterable types (i.e., list or tuple)
            for k, v in config.items():
                if not isinstance(v, list) or isinstance(v, tuple):
                    config[k] = [v]

            # We treat num_clients differently to enable early stopping
            num_clients = config.pop("num_clients")

            # Generate all possible combinations of prompt config values
            for vals in itertools.product(*[config[k] for k in prompt_config_keys]):
                config_product = {k: v for k, v in zip(prompt_config_keys, vals)}
                config_product["num_clients"] = num_clients
                all_config_product.append(config_product)

        logger.info(f"Generated {len(all_config_product)} benchmark run setting(s)")

        for config in all_config_product:
            num_clients = config.pop("num_clients")
            prompt_config = PromptConfig(**config)
            yield num_clients, prompt_config

    def _generate_requests(self, prompt_config: PromptConfig, num_clients: int) -> None:
        logger.info("Generating Prompts")

        warmup_prompts = self.config.warmup_requests * num_clients
        workload_prompts = max(
            self.config.min_requests, self.config.num_requests_per_client * num_clients
        )
        for prompt in self.prompt_generator(
            config=prompt_config, num_prompts=warmup_prompts + workload_prompts
        ):
            self.client_launcher.add_request(prompt)

        logger.info(
            f"Generated {warmup_prompts} warmup and {workload_prompts} workload prompts."
        )

    def _get_output_dir(self) -> Path:
        return self.config.result_dir / self.config.api / self.config.model

    def _get_output_path(self, prompt_config: PromptConfig, num_clients: int) -> Path:
        output_dir = self._get_output_dir()
        output_file = f"prompt{prompt_config.prompt_length}_gen{prompt_config.max_new_tokens}_clients{num_clients}.json"
        return output_dir / output_file

    def _process_responses(
        self, prompt_config: PromptConfig, num_clients: int
    ) -> List[Response]:
        output_path = self._get_output_path(
            prompt_config=prompt_config, num_clients=num_clients
        )

        logger.info(f"Saving results to {output_path}")

        all_responses = []
        while True:
            try:
                all_responses.append(self.client_launcher.get_response())
            except queue.Empty:
                break

        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump([r.to_dict() for r in all_responses], fh, indent=2)

        logger.info(f"Saved {len(all_responses)} responses to {output_path}")

        return all_responses

    def _print_result_summary(
        self, all_responses: List[Response], num_clients: int
    ) -> None:
        num_responses = int(len(all_responses))
        mean_latency = sum([r.request_time for r in all_responses]) / num_responses
        query_throughput = num_clients / mean_latency
        mean_prompt_length = int(
            sum([r.prompt_tokens for r in all_responses]) / num_responses
        )
        mean_gen_length = int(
            sum([r.generated_tokens for r in all_responses]) / num_responses
        )
        logger.info(
            f"Result summary - # Requests: {num_responses:d}, Mean Prompt Length: {mean_prompt_length:d} tokens, Mean Generation Length: {mean_gen_length:d} tokens, Mean Latency: {mean_latency:.2f} s, Throughput: {query_throughput:.2f} queries/s,"
        )

    def _check_early_stop(self, all_responses: List[Response]) -> bool:
        if not all_responses:
            return False
        mean_latency = sum([r.request_time for r in all_responses]) / len(all_responses)
        if mean_latency >= self.config.early_stop_latency:
            logger.info(
                f"Mean latency of {mean_latency:.2f} exceeds early stopping threshold of {self.config.early_stop_latency}. Stopping early."
            )
            return True
        return False

    def _skip_existing_result(
        self, prompt_config: PromptConfig, num_clients: int
    ) -> bool:
        output_path = self._get_output_path(
            prompt_config=prompt_config, num_clients=num_clients
        )
        if output_path.exists():
            if self.config.force:
                logger.info(
                    f"Result already exists, but force flag is set. Overwriting benchmark with {num_clients} client(s) and prompt config: {prompt_config}"
                )
                return False
            else:
                logger.info(
                    f"Result already exists, skipping benchmark with {num_clients} client(s) and prompt config: {prompt_config}"
                )
                return True
        return False

    def run(self) -> None:
        # Start the client service
        self.client_launcher.start_service()

        # Generate all benchmark settings from user config(s)
        for num_clients_list, prompt_config in self._benchmark_settings():
            all_responses = []
            for num_clients in sorted(num_clients_list):
                if self._skip_existing_result(
                    prompt_config=prompt_config, num_clients=num_clients
                ):
                    continue

                if self._check_early_stop(all_responses):
                    break

                logger.info(
                    f"Running benchmark with {num_clients} client(s) and prompt config: {prompt_config}"
                )
                # Clear out queues and generate request prompts
                self.client_launcher.clear_queues()
                self._generate_requests(
                    prompt_config=prompt_config, num_clients=num_clients
                )

                # Launch the clients and process requests
                self.client_launcher.run_parallel_clients(num_clients=num_clients)

                # Process raw responses and save results to file
                all_responses = self._process_responses(
                    prompt_config=prompt_config, num_clients=num_clients
                )

                self._print_result_summary(
                    all_responses=all_responses, num_clients=num_clients
                )

        # Stop the client service
        self.client_launcher.stop_service()


if __name__ == "__main__":
    from .arg_parsing import parse_args_to_configs

    benchmark_config, client_config = parse_args_to_configs(sys.argv[1:])
    benchmark_runner = BenchmarkRunner(benchmark_config, client_config)
    benchmark_runner.run()
