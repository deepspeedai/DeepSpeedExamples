# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
import json
import re
import os
from tabulate import tabulate
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from statistics import mean
from typing import List
from collections import defaultdict

import numpy as np
from transformers import AutoTokenizer


tokenizer = None


@dataclass
class ResponseDetails:
    generated_tokens: List[str]
    prompt: str
    start_time: float
    end_time: float
    model_time: float
    token_gen_time: List[float]


@dataclass
class ProfilingSummary:
    throughput: float
    latency: float
    token_gen_latency: float
    first_token_latency: float
    tokens_per_sec: float


def parse_args():
    parser = argparse.ArgumentParser(description="Postprocess results")
    parser.add_argument("-i", "--input_path", type=Path, default="results.json")

    args = parser.parse_args()
    return args


def get_tokenizer(model=None):
    global tokenizer
    if tokenizer is None:
        if model==None:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model)
    return tokenizer


def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    args = data["args"]

    response_details = []
    for response in data["response_details"]:
        response_details.append(ResponseDetails(**response))

    return args, response_details


def get_summary(args, response_details):
    num_clients = args["num_clients"]

    # Calculate latency and throughput using P95 latency
    latency = mean([r.end_time - r.start_time for r in response_details])
    throughput = num_clients / latency

    tokens_per_sec = mean(
        [
            (len(get_tokenizer(args["model"]).tokenize(r.prompt)) +
            len(get_tokenizer(args["model"]).tokenize(r.generated_tokens)) if type(r.generated_tokens) == str
            else len(r.generated_tokens))
            / (r.end_time - r.start_time)
            for r in response_details
        ]
    )

    # For non-streaming results, we don't have any token_gen_time information
    first_token_latency = 0.0
    token_gen_latency = 0.0
    if response_details[0].token_gen_time:
        first_token_latency = mean([r.token_gen_time[0] for r in response_details])
        token_gen_latency_flat = reduce(
            list.__add__,
            [
                r.token_gen_time[1:-1]
                for r in response_details
                if len(r.token_gen_time) > 2
            ],
        )
        token_gen_latency = mean([t for t in token_gen_latency_flat])

    return ProfilingSummary(
        throughput, latency, token_gen_latency, first_token_latency, tokens_per_sec
    )


def get_token_latency(
    response_details, percentile=None, variance=False, cumulative=False
):
    req_latencies = [r.token_gen_time for r in response_details]
    if cumulative:
        req_latencies = [
            np.cumsum(np.array(r.token_gen_time)).tolist() for r in response_details
        ]
    max_gen_length = max([len(r.generated_tokens) for r in response_details])
    latency = []
    for i in range(max_gen_length):
        if variance:
            token_latency_step = np.var(
                [latency[i] for latency in req_latencies if len(latency) > i]
            )
        if percentile is None:
            token_latency_step = [
                latency[i] for latency in req_latencies if len(latency) > i
            ]
        else:
            token_latency_step = np.percentile(
                [latency[i] for latency in req_latencies if len(latency) > i],
                percentile,
            )

        latency.append(token_latency_step)

    return latency


def get_token_acc_latency(response_details, percentile=99):
    return get_token_latency(response_details, percentile, cumulative=True)


if __name__ == "__main__":
    args = parse_args()
    prof_args, response_details = read_json(args.input_path)

    ps = get_summary(prof_args, response_details)
    print(
        f"Deployment: {prof_args['deployment_name']} Clients: {prof_args['num_clients']}, "
        + f"Query throughput: {ps.throughput:.3f} queries/s, "
        + f"Token throughput (total): {ps.tokens_per_sec:.3f} tokens/s, "
        + f"Query latency: {ps.latency:.3f} s, "
        + f"Token generation latency: {ps.token_gen_latency:.3f} s/token, "
        + f"First token received: {ps.first_token_latency:.3f} s"
    )

def get_result_sets(args: argparse.Namespace) -> set():
    result_params = None
    result_re = re.compile(
        r"(.+)-tp(\d+)-bs(\d+)-replicas(\d+)-prompt(\d+)-gen(\d+)-clients.*.json"
    )

    data_sets = defaultdict(set)

    if hasattr(args, "data_dirs"):
        data_set_dirs = args.data_dirs
    elif hasattr(args, "backend"):
        data_set_dirs = args.backend

    # Generate data sets
    for data in data_set_dirs:
        if hasattr(args, "log_dir"):
            os_path = os.path.join(args.log_dir, data)
        else:
            os_path = os.path.join(data)

        for f in os.listdir(os_path):
            match = result_re.match(f)
            if match:
                data_sets[data].add(match.groups())

    # Intersection between all sets
    for data_set in data_sets.values():
        if result_params == None:
            result_params = data_set
        else:
            result_params = result_params.intersection(data_set)

    # Warning messages about skipped sets
    for key, data_set in data_sets.items():
        difference = data_set.difference(result_params)
        if difference:
            print(f"WARNING: data {key} has result combinations that are not present in all data sets:")
            print(tabulate(difference, headers=["model", "tp_size", "bs", "replicas", "prompt", "gen"]))
            print("")

    return result_params
