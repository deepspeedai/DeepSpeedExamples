# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import glob
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import numpy as np

from .postprocess_results import read_json, get_summary

bs = 768

REPLICA_NUMS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

tp_sizes = {
    "70b": [4],
}

prompt_gen_pairs = [
    (2600, 60),
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=Path, default=".")
    parser.add_argument("--out_dir", type=Path, default="charts/repl_scale")
    args = parser.parse_args()
    return args


def extract_values(file_pattern):
    files = glob.glob(file_pattern)

    clients = []
    throughputs = []
    latencies = []
    for f in files:
        prof_args, response_details = read_json(f)
        summary = get_summary(prof_args, response_details)
        clients.append(prof_args["client_num"])
        throughputs.append(summary.throughput)
        latencies.append(summary.latency)

    return clients, throughputs, latencies


def output_charts(model_size, tp, bs, prompt, gen, log_dir, out_dir):
    if not log_dir.exists():
        print(f"Log directory {log_dir} does not exist")
        return

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    throughputs = {}
    for repl in REPLICA_NUMS:
        mii_file_pattern = f"{log_dir}/logs.llama2-{model_size}-tp{tp}-b{bs}_repl{repl}/llama2-{model_size}-tp{tp}-b{bs}_repl{repl}_c*_p{prompt}_g{gen}.json"
        print(f"Looking for {mii_file_pattern}")
        clients, mii_throughputs, mii_latencies = extract_values(mii_file_pattern)

        for c, th in zip(clients, mii_throughputs):
            client_per_repl = c // repl
            if client_per_repl not in throughputs:
                throughputs[client_per_repl] = []
            print(f"Throughput for {client_per_repl} clients: {th}")
            throughputs[client_per_repl].append(th)

    for c in throughputs:

        # Plotting the scatter plot
        plt.figure(figsize=(6, 4))

        plt.bar(REPLICA_NUMS, throughputs[c], color="blue", alpha=0.9)

        fit_x_list = np.arange(min(REPLICA_NUMS), max(REPLICA_NUMS), 0.1)
        mii_fit_model = np.polyfit(REPLICA_NUMS, throughputs[c], 1)
        mii_model_fn = np.poly1d(mii_fit_model)
        plt.plot(fit_x_list, mii_model_fn(fit_x_list), color="blue", linestyle="--")

        plt.title(
            f"Model Llama 2 {model_size.upper()}, Prompt: {prompt}, Generation: {gen}, TP: {tp}"
        )
        plt.xlabel("Number of replicas", fontsize=14)
        plt.ylabel("Throughput (queries/s)", fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        out_file = out_dir / f"repl_scale_llama{model_size}_tp{tp}_p{prompt}g{gen}.png"
        plt.savefig(out_file)


if __name__ == "__main__":
    raise NotImplementedError("This script is not up to date")
    args = get_args()

    for model_size, tps in tp_sizes.items():
        for tp in tps:
            for prompt, gen in prompt_gen_pairs:
                output_charts(
                    model_size, tp, bs, prompt, gen, args.log_dir, args.out_dir
                )
