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

tp_sizes = {
    # "7b": [1],
    "13b": [1, 2, 4],
    # "70b": [4, 8],
}

prompt_gen_pairs = [
    (1200, 60),
    (1200, 128),
    (2600, 60),
    (2600, 128),
    (2600, 256),
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=Path, default="logs.release")
    parser.add_argument("--out_dir", type=Path, default="charts/tp_sizes")
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
        clients.append(prof_args["client_num"])
        throughputs.append(summary.throughput)
        latencies.append(summary.latency)

    return clients, throughputs, latencies


def output_charts(model_size, tps, bs, prompt, gen, log_dir, out_dir):
    if not log_dir.exists():
        print(f"Log directory {log_dir} does not exist")
        return

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Plotting the scatter plot
    plt.figure(figsize=(6, 4))

    colors = ["orange", "green", "brown"]

    for tp, color in zip(tps, colors):
        mii_file_pattern = f"{log_dir}/logs.llama2-{model_size}-tp{tp}-b{bs}/llama2-{model_size}-tp{tp}-b{bs}_c*_p{prompt}_g{gen}.json"
        _, mii_throughputs, mii_latencies = extract_values(mii_file_pattern)

        if len(mii_throughputs) == 0:
            continue

        n_params = int(model_size[:-1])
        tflops_per_query = n_params * (prompt + gen) * 2 * 1e-3
        mii_tflops = [th * tflops_per_query / tp for th in mii_throughputs]

        plt.scatter(
            mii_tflops, mii_latencies, label=f"TP={tp}", marker="o", color=color
        )
        fit_mii_x_list = np.arange(min(mii_tflops), max(mii_tflops), 0.01)
        mii_fit_model = np.polyfit(mii_tflops, mii_latencies, 3)
        mii_model_fn = np.poly1d(mii_fit_model)
        plt.plot(
            fit_mii_x_list,
            mii_model_fn(fit_mii_x_list),
            color=color,
            alpha=0.5,
            linestyle="--",
        )

    plt.title(
        f"Model Llama 2 {model_size.upper()}, Prompt: {prompt}, Generation: {gen}, TP: {tps}"
    )
    plt.xlabel("TFLOPs (per GPU)", fontsize=14)
    plt.ylabel("Latency", fontsize=14)
    plt.legend()
    plt.grid(True)
    # plt.show()
    out_file = (
        out_dir
        / f"tp_sizes_llama{model_size}_tp{'_'.join([str(tp) for tp in tps])}_p{prompt}g{gen}.png"
    )
    plt.savefig(out_file)


if __name__ == "__main__":
    raise NotImplementedError("This script is not up to date")
    args = get_args()

    for model_size, tps in tp_sizes.items():
        for prompt, gen in prompt_gen_pairs:
            output_charts(model_size, tps, bs, prompt, gen, args.log_dir, args.out_dir)
