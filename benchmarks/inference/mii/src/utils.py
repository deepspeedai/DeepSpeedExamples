# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
import copy
import itertools
import json
import os

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator, List

try:
    from .defaults import ARG_DEFAULTS, MODEL_DEFAULTS
    from .postprocess_results import get_summary, ResponseDetails
except ImportError:
    from defaults import ARG_DEFAULTS, MODEL_DEFAULTS
    from postprocess_results import get_summary, ResponseDetails

# For these arguments, users can provide multiple values when running the
# benchmark. The benchmark will iterate over all possible combinations.
SERVER_PARAMS = ["tp_size", "max_ragged_batch_size", "num_replicas"]
CLIENT_PARAMS = ["mean_prompt_length", "mean_max_new_tokens", "num_clients"]

AML_REQUIRED_PARAMS = ["aml_api_url", "aml_api_key", "deployment_name", "model"]


def parse_args(
    server_args: bool = False, client_args: bool = False
) -> argparse.Namespace:
    if not (server_args or client_args):
        raise ValueError("Must specify server_args or client_args or both")

    # Server args
    server_parser = argparse.ArgumentParser(add_help=False)
    server_parser.add_argument(
        "--tp_size", type=int, nargs="+", default=None, help="Tensor parallelism size"
    )
    server_parser.add_argument(
        "--max_ragged_batch_size",
        type=int,
        nargs="+",
        default=None,
        help="Max batch size for ragged batching",
    )
    server_parser.add_argument(
        "--num_replicas",
        type=int,
        nargs="+",
        default=None,
        help="Number of FastGen model replicas",
    )
    server_parser.add_argument(
        "cmd",
        type=str,
        nargs="?",
        choices=["start", "stop", "restart"],
        help="Command for running server.py to manually start/stop/restart a server",
    )
    server_parser.add_argument(
        "--client_only", action="store_true", help="Run client only with server started"
    )


    # Client args
    client_parser = argparse.ArgumentParser(add_help=False)
    client_parser.add_argument(
        "--max_prompt_length", type=int, default=None, help="Max length a prompt can be"
    )
    client_parser.add_argument(
        "--mean_prompt_length",
        type=int,
        nargs="+",
        default=None,
        help="Mean prompt length in tokens",
    )
    client_parser.add_argument(
        "--mean_max_new_tokens",
        type=int,
        nargs="+",
        default=None,
        help="Mean number of new tokens to generate per prompt",
    )
    client_parser.add_argument(
        "--num_clients",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        help="Number of concurrent clients",
    )
    client_parser.add_argument(
        "--num_requests",
        type=int,
        default=None,
        help="Number of requests to process by clients",
    )
    client_parser.add_argument(
        "--prompt_length_var", type=float, default=0.3, help="Variance of prompt length"
    )
    client_parser.add_argument(
        "--max_new_tokens_var",
        type=float,
        default=0.3,
        help="Variance of max new tokens",
    )
    client_parser.add_argument(
        "--warmup", type=int, default=1, help="Number of warmup requests to process"
    )
    client_parser.add_argument(
        "--use_thread", action="store_true", help="Use threads instead of processes"
    )
    client_parser.add_argument(
        "--stream", action="store_true", help="Stream generated tokens"
    )
    client_parser.add_argument(
        "--out_json_dir",
        type=Path,
        default="./results/",
        help="Directory to save result JSON files",
    )
    client_parser.add_argument(
        "--openai_api_url",
        type=str,
        default=None,
        help="When using the openai API backend, this is the API URL that points to an openai api server",
    )
    client_parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="When using the openai API backend, this is the API key for a given openai_api_url",
    )
    client_parser.add_argument(
        "--aml_api_url",
        type=str,
        default=None,
        help="When using the AML backend, this is the API URL that points to an AML endpoint",
    )
    client_parser.add_argument(
        "--aml_api_key",
        type=str,
        default=None,
        help="When using the AML backend, this is the API key for a given aml_api_url",
    )

    # Create the parser, inheriting from the server and/or client parsers
    parents = []
    if server_args:
        parents.append(server_parser)
    if client_args:
        parents.append(client_parser)

    # Common args
    parser = argparse.ArgumentParser(parents=parents)
    parser.add_argument(
        "--model", type=str, default=None, help="HuggingFace.co model name"
    )
    parser.add_argument(
        "--deployment_name",
        type=str,
        default=None,
        help="When using FastGen backend, specifies which model deployment to use. When using AML backend, specifies the name of the deployment",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["aml", "fastgen", "vllm", "openai"],
        default="fastgen",
        help="Which backend to benchmark",
    )
    parser.add_argument(
        "--overwrite_results", action="store_true", help="Overwrite existing results"
    )
    parser.add_argument("--fp6", action="store_true", help="Enable FP6")

    # Parse arguments
    args = parser.parse_args()

    # Verify that AML required parameters are defined before filling in defaults
    if args.backend == "aml":
        for k in AML_REQUIRED_PARAMS:
            if getattr(args, k) is None:
                raise ValueError(f"AML backend requires {k} to be specified")

    # Set default values for model-specific parameters
    if args.model in MODEL_DEFAULTS:
        for k, v in MODEL_DEFAULTS[args.model].items():
            if hasattr(args, k) and getattr(args, k) is None:
                setattr(args, k, v)

    # Grab any remaining default values not specified for a model
    for k, v in ARG_DEFAULTS.items():
        if hasattr(args, k) and getattr(args, k) is None:
            setattr(args, k, v)

    # If we are not running the benchmark, we need to make sure to only have one
    # value for the server args
    if server_args and not client_args:
        for k in SERVER_PARAMS:
            if not isinstance(getattr(args, k), int):
                setattr(args, k, getattr(args, k)[0])

    return args


def get_args_product(
    args: argparse.Namespace, which: List[str] = None
) -> Iterator[argparse.Namespace]:
    if which is None:
        return copy.deepcopy(args)
    for k in which:
        if isinstance(getattr(args, k), int):
            setattr(args, k, [getattr(args, k)])
    arg_values_product = itertools.product(*[getattr(args, k) for k in which])
    for arg_values in arg_values_product:
        args_copy = copy.deepcopy(args)
        for k, v in zip(which, arg_values):
            setattr(args_copy, k, v)
        yield args_copy


def get_results_path(args: argparse.Namespace) -> Path:
    return Path(
        f"{args.out_json_dir}_{args.backend}/",
        "-".join(
            (
                args.model.replace("/", "_"),
                f"tp{args.tp_size}",
                f"bs{args.max_ragged_batch_size}",
                f"replicas{args.num_replicas}",
                f"prompt{args.mean_prompt_length}",
                f"gen{args.mean_max_new_tokens}",
                f"clients{args.num_clients}",
            )
        )
        + ".json",
    )


def print_summary(
    args: argparse.Namespace, response_details: List[ResponseDetails]
) -> None:
    ps = get_summary(vars(args), response_details)
    print(
        f"Deployment: {args.deployment_name} Clients: {args.num_clients}, "
        + f"Prompt (mean): {args.mean_prompt_length} tokens, "
        + f"Generation (mean): {args.mean_max_new_tokens} tokens, "
        + f"Query throughput: {ps.throughput:.3f} queries/s, "
        + f"Token throughput (total): {ps.tokens_per_sec:.3f} tokens/s, "
        + f"Query latency: {ps.latency:.3f} s, "
        + f"Token generation latency: {ps.token_gen_latency:.3f} s/token, "
        + f"First token received: {ps.first_token_latency:.3f} s"
    )


def save_json_results(
    args: argparse.Namespace, response_details: List[ResponseDetails]
) -> None:
    args_dict = vars(args)
    # Remove AML key from args dictionary
    if "aml_api_key" in args_dict:
        args_dict["aml_api_key"] = None
    out_json_path = get_results_path(args)
    os.makedirs(out_json_path.parent, exist_ok=True)

    with open(out_json_path, "w") as f:
        args_dict["out_json_dir"] = str(out_json_path)  # Path is not JSON serializable
        data = {
            "args": args_dict,
            "time": str(datetime.now()),
            "response_details": [asdict(r) for r in response_details],
        }
        json.dump(data, f, indent=2)


def results_exist(args: argparse.Namespace) -> bool:
    return get_results_path(args).exists()
