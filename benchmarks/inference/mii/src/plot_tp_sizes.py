# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import glob
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import numpy as np
import re
from collections import defaultdict

from postprocess_results import read_json, get_summary, get_result_sets

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=["aml", "fastgen", "vllm"], default=["aml", "fastgen", "vllm"], \
                        nargs=1, help="Specify the single backend to generate plots for")
    parser.add_argument("--log_dir", type=Path, default="logs.release")
    parser.add_argument("--out_dir", type=Path, default="./plots/tp_sizes")
    args = parser.parse_args()
    return args


def extract_values(file_pattern):
    files = glob.glob(file_pattern)

    print(f"Found {len(files)}")
    print("\n".join(files))

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


def output_charts(args, model, tp_list, bs, replicas, prompt, gen, log_dir, out_dir):
    if not log_dir.exists():
        print(f"Log directory {log_dir} does not exist")
        return

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Plotting the scatter plot
    plt.figure()

    for tp in tp_list:
        result_file_pattern = f"{model}-tp{tp}-bs{bs}-replicas{replicas}-prompt{prompt}-gen{gen}-clients*.json"
        file_pattern = f"{log_dir}/{args.backend[0]}/{result_file_pattern}"
        _, throughputs, latencies = extract_values(file_pattern)

        if len(throughputs) == 0:
            continue

        model_size = re.match('.*?(\d+[b|B|m|M])', model).groups()[0]
        n_params = int(model_size[:-1])
        if model_size[-1].lower() == 'm':
            # Scale n_params approriately for millions
            n_params = n_params / 1000
        tflops_per_query = n_params * (int(prompt) + int(gen)) * 2 * 1e-3
        tflops = [th * tflops_per_query / tp for th in throughputs]

        plt.scatter(
            tflops, latencies, label=f"TP={tp}", marker="o"
        )
        fit_x_list = np.arange(min(tflops), max(tflops), 0.01)
        fit_model = np.polyfit(tflops, latencies, 3)
        model_fn = np.poly1d(fit_model)
        plt.plot(
            fit_x_list,
            model_fn(fit_x_list),
            alpha=0.5,
            linestyle="--",
        )

    plt.title(
        f"Model: {model}, Prompt: {prompt}, Generation: {gen}, TP: {tp_list}\n\
        Replicas: {replicas}, Backend: {args.backend[0]}"
    )
    plt.xlabel("TFLOPs (per GPU)", fontsize=14)
    plt.ylabel("Latency (s)", fontsize=14)
    plt.legend()
    plt.grid(True)
    out_file = (
        out_dir
        / f"tp_sizes_{model}_tp{'_'.join([str(tp) for tp in tp_list])}_p{prompt}g{gen}r{replicas}.png"
    )
    plt.savefig(out_file)


if __name__ == "__main__":
    args = get_args()

    tp_sets = defaultdict(lambda: defaultdict(set))
    result_params = get_result_sets(args)

    # Find all tp_sizes across same sets
    for model, tp_size, bs, replicas, prompt, gen in result_params:
        key = f'{model}_{bs}_{replicas}_{prompt}_{gen}'
        tp_sets[key]['config'].add((model, bs, replicas, prompt, gen))
        tp_sets[key]['tp_list'].add(int(tp_size))

    for tp_set in tp_sets.values():
        for model, bs, replicas, prompt, gen in tp_set['config']:
            tp_list = sorted(tp_set['tp_list'])
            output_charts(
                args=args,
                model=model,
                tp_list=tp_list,
                bs=bs,
                replicas=replicas,
                prompt=prompt,
                gen=gen,
                log_dir=args.log_dir,
                out_dir=args.out_dir,
            )
