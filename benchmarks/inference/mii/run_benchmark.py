# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from src.client import run_client
from src.server import start_server, stop_server
from src.utils import (
    get_args_product,
    parse_args,
    print_summary,
    results_exist,
    save_json_results,
    CLIENT_PARAMS,
    SERVER_PARAMS,
)


def run_benchmark() -> None:
    args = parse_args(server_args=True, client_args=True)

    for server_args in get_args_product(args, which=SERVER_PARAMS):
        if server_args.backend != "aml" and not server_args.client_only:
            start_server(server_args)

        for client_args in get_args_product(server_args, which=CLIENT_PARAMS):
            if results_exist(client_args) and not args.overwrite_results:
                print(
                    f"Found existing results and skipping current setting. To ignore existing results, use --overwrite_results"
                )
                continue

            if client_args.num_requests is None:
                client_args.num_requests = client_args.num_clients * 4 + 32
            response_details = run_client(client_args)
            print_summary(client_args, response_details)
            save_json_results(client_args, response_details)

        if server_args.backend != "aml" and not server_args.client_only:
            stop_server(server_args)


if __name__ == "__main__":
    run_benchmark()
