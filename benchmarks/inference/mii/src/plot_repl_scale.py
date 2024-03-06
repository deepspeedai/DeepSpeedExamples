# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import glob
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import numpy as np
from collections import defaultdict

from postprocess_results import read_json, get_summary, get_result_sets

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=["fastgen"], default=["fastgen"], \
                        nargs=1, help="Specify the single backend to generate plots for")
    parser.add_argument("--clients_per_replica", type=int, required=False, default=None, help="Optional \
                        argument to specify explicit clients/replica to generate plot for")
    parser.add_argument("--log_dir", type=Path, default="./results")
    parser.add_argument("--out_dir", type=Path, default="./plots/repl_scale")
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
        clients.append(prof_args["num_clients"])
        throughputs.append(summary.throughput)
        latencies.append(summary.latency)

    return clients, throughputs, latencies


def output_charts(args, model, tp_size, bs, replica_nums, prompt, gen, log_dir, out_dir):
    if not log_dir.exists():
        print(f"Log directory {log_dir} does not exist")
        return

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    throughputs = {}
    for repl in replica_nums:
        result_file_pattern = f"{model}-tp{tp_size}-bs{bs}-replicas{repl}-prompt{prompt}-gen{gen}-clients*.json"
        mii_file_pattern = f"{log_dir}/fastgen/{result_file_pattern}"
        print(f"Looking for {mii_file_pattern}")
        clients, mii_throughputs, mii_latencies = extract_values(mii_file_pattern)

        for c, th in zip(clients, mii_throughputs):
            client_per_repl = c // repl
            if client_per_repl not in throughputs:
                throughputs[client_per_repl] = []
            print(f"Throughput for {client_per_repl} clients: {th}")
            throughputs[client_per_repl].append(th)

    for c in throughputs:
        if args.clients_per_replica != None and args.clients_per_replica != c:
            continue
        if len(throughputs[c]) == len(replica_nums):
            print(f"Generating figure for {c} clients/replica.")
            # Plotting the scatter plot
            plt.figure()

            plt.bar(replica_nums, throughputs[c], color="blue", alpha=0.9)

            fit_x_list = np.arange(min(replica_nums), max(replica_nums), 0.1)
            mii_fit_model = np.polyfit(replica_nums, throughputs[c], 1)
            mii_model_fn = np.poly1d(mii_fit_model)
            plt.plot(fit_x_list, mii_model_fn(fit_x_list), color="blue", linestyle="--")

            plt.title(
                f"Model: {model}, Prompt: {prompt}, Generation: {gen}\n\
                TP: {tp_size}, Clients/Replica: {c}"
            )
            plt.xlabel("Number of replicas", fontsize=14)
            plt.ylabel("Throughput (queries/s)", fontsize=14)
            plt.grid(True)
            plt.tight_layout()
            out_file = out_dir / f"repl_scale_{model}_tp{tp_size}_p{prompt}g{gen}_c_per_r{c}.png"
            plt.savefig(out_file)


if __name__ == "__main__":
    args = get_args()

    replica_sets = defaultdict(lambda: defaultdict(set))
    result_params = get_result_sets(args)

    # Find all replicas across same sets
    for model, tp_size, bs, replicas, prompt, gen in result_params:
        key = f'{model}_{tp_size}_{bs}_{prompt}_{gen}'
        replica_sets[key]['config'].add((model, tp_size, bs, prompt, gen))
        replica_sets[key]['replicas'].add(int(replicas))

    for replica_set in replica_sets.values():
        for model, tp_size, bs, prompt, gen in replica_set['config']:
            replica_nums = sorted(replica_set['replicas'])
            output_charts(
                args=args,
                model=model,
                tp_size=tp_size,
                bs=bs,
                replica_nums=replica_nums,
                prompt=prompt,
                gen=gen,
                log_dir=args.log_dir,
                out_dir=args.out_dir,
            )
