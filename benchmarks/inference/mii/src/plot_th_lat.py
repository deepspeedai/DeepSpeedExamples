# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from postprocess_results import read_json, get_summary


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=Path, default="./results")
    parser.add_argument("--out_dir", type=Path, default="./plots/throughput_latency")
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


def output_charts(model, tp_size, bs, replicas, prompt, gen, log_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    result_file_pattern = f"{model}-tp{tp_size}-bs{bs}-replicas{replicas}-prompt{prompt}-gen{gen}-clients*.json"
    mii_file_pattern = f"{log_dir}/fastgen/{result_file_pattern}"
    vllm_file_pattern = f"{log_dir}/vllm/{result_file_pattern}"

    _, mii_throughputs, mii_latencies = extract_values(mii_file_pattern)
    _, vllm_throughputs, vllm_latencies = extract_values(vllm_file_pattern)

    # Plotting the scatter plot
    plt.figure(figsize=(6, 4))

    if len(vllm_throughputs) > 0:
        plt.scatter(
            vllm_throughputs, vllm_latencies, label=f"vLLM", marker="x", color="orange"
        )
        fit_vllm_x_list = np.arange(min(vllm_throughputs), max(vllm_throughputs), 0.01)
        vllm_vllm_model = np.polyfit(vllm_throughputs, vllm_latencies, 3)
        vllm_model_fn = np.poly1d(vllm_vllm_model)
        plt.plot(
            fit_vllm_x_list,
            vllm_model_fn(fit_vllm_x_list),
            color="orange",
            alpha=0.5,
            linestyle="--",
        )

    plt.scatter(
        mii_throughputs,
        mii_latencies,
        label=f"DeepSpeed FastGen",
        marker="o",
        color="blue",
    )
    fit_mii_x_list = np.arange(min(mii_throughputs), max(mii_throughputs), 0.01)
    mii_fit_model = np.polyfit(mii_throughputs, mii_latencies, 3)
    mii_model_fn = np.poly1d(mii_fit_model)
    plt.plot(
        fit_mii_x_list,
        mii_model_fn(fit_mii_x_list),
        color="blue",
        alpha=0.5,
        linestyle="--",
    )

    plt.title(f"Model {model}, Prompt: {prompt}, Generation: {gen}, TP: {tp_size}")
    plt.xlabel("Throughput (queries/s)", fontsize=14)
    plt.ylabel("Latency", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_file = (
        out_dir
        / f"{model}-tp{tp_size}-bs{bs}-replicas{replicas}-prompt{prompt}-gen{gen}.png"
    )
    print(f"Saving {out_file}")
    plt.savefig(out_file)


if __name__ == "__main__":
    args = get_args()

    if not args.log_dir.exists():
        raise ValueError(f"Log dir {args.log_dir} does not exist")

    result_params = set()
    result_re = re.compile(
        r"(.+)-tp(\d+)-bs(\d+)-replicas(\d+)-prompt(\d+)-gen(\d+)-clients.*.json"
    )
    for f in os.listdir(os.path.join(args.log_dir, "fastgen")):
        match = result_re.match(f)
        if match:
            result_params.add(match.groups())

    for model, tp_size, bs, replicas, prompt, gen in result_params:
        output_charts(
            model=model,
            tp_size=tp_size,
            bs=bs,
            replicas=replicas,
            prompt=prompt,
            gen=gen,
            log_dir=args.log_dir,
            out_dir=args.out_dir,
        )
