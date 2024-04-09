# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
import glob
import os
import re
import yaml
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from postprocess_results import read_json, get_summary, get_result_sets


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", type=str, nargs="+", \
                        help="Specify the data directories to generate plots for")
    parser.add_argument("--out_dir", type=Path, default="./plots/throughput_latency")
    parser.add_argument("--model_name", type=str, default="", help="Optional model name override")
    args = parser.parse_args()
    return args


def extract_values(file_pattern):
    files = glob.glob(file_pattern)

    print(f"Found {len(files)}")
    print("\n".join(files))

    clients = []
    throughputs = []
    latencies = []
    extra_args = {}
    for f in files:
        prof_args, response_details = read_json(f)
        summary = get_summary(prof_args, response_details)
        clients.append(prof_args["num_clients"])
        throughputs.append(summary.throughput)
        latencies.append(summary.latency)

    return clients, throughputs, latencies, prof_args


def output_charts(model, tp_size, bs, replicas, prompt, gen, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    result_file_pattern = f"{model}-tp{tp_size}-bs{bs}-replicas{replicas}-prompt{prompt}-gen{gen}-clients*.json"

    plt.figure()

    for data_dir in args.data_dirs:
        file_pattern = f"{data_dir}/{result_file_pattern}"
        _, throughputs, latencies, prof_args = extract_values(file_pattern)

        kwargs = {}
        kwargs["label"] = str(data_dir)
        kwargs["marker"] = "o"
        kwargs["linestyle"] = "--"

        fit_kwargs = {}
        fit_kwargs["linestyle"] = "--"
        plot_fit_line = True

        polyfit_degree = 3
        plot_fn = plt.scatter

        plot_config = glob.glob(f"{data_dir}/plot_config.yaml")

        latencies = sorted(latencies)
        throughputs = sorted(throughputs)

        if plot_config:
            plot_config = plot_config[0]
            plot_config = yaml.safe_load(Path(plot_config).read_text())
            plot_keys = plot_config.keys()

            # If x_max specified, clip data
            if "x_max" in plot_keys:
                for i, throughput in enumerate(throughputs):
                    if throughput > plot_config["x_max"]:
                        latencies = latencies[:i]
                        throughputs = throughputs[:i]
                        break

            # If y_max specified, clip data
            if "y_max" in plot_keys:
                for i, latency in enumerate(latencies):
                    if latency > plot_config["y_max"]:
                        latencies = latencies[:i]
                        throughputs = throughputs[:i]
                        break

            # Set polyfit degree
            polyfit_degree = plot_config.get("polyfit_degree", polyfit_degree)

            # Select plot type
            if polyfit_degree == 0:
                plot_fit_line = False

            # Main plot kwargs
            if "label" in plot_keys:
                kwargs["label"] = plot_config["label"]
            if "marker" in plot_keys:
                kwargs["marker"] = plot_config["marker"]
            if "color" in plot_keys:
                kwargs["color"] = plot_config["color"]
            if "linestyle" in plot_keys:
                kwargs["linestyle"] = plot_config["linestyle"]

            # Fit line kwargs
            if "color" in plot_keys:
                fit_kwargs["color"] = plot_config["color"]
            if "linestyle" in plot_keys:
                fit_kwargs["linestyle"] = plot_config["linestyle"]

        if len(throughputs) > 0:
            plot = plot_fn(
                throughputs,
                latencies,
                **kwargs,
            )

            if plot_fn == plt.plot:
                plot_color = plot[0].get_color()
            else:
                plot_color = plot.get_facecolor()[0]

            if not "color" in fit_kwargs.keys():
                fit_kwargs["color"] = plot_color

            fit_x_list = np.arange(min(throughputs), max(throughputs), 0.01)
            data_model = np.polyfit(throughputs, latencies, polyfit_degree)
            model_fn = np.poly1d(data_model)
            x = fit_x_list if plot_fit_line else throughputs
            y = model_fn(fit_x_list) if plot_fit_line else latencies
            plt.plot(
                x,
                y,
                alpha=0.5,
                **fit_kwargs,
            )

    # Generic plot formatting
    if args.model_name:
        model_label = args.model_name
    else:
        model_label = model

    plt.title(f"Model: {model_label}, Prompt: {prompt}, Generation: {gen}, TP: {tp_size}")
    plt.xlabel("Throughput (queries/s)", fontsize=14)
    plt.ylabel("Latency (s)", fontsize=14)
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

    result_params = get_result_sets(args)

    for model, tp_size, bs, replicas, prompt, gen in result_params:
        output_charts(
            model=model,
            tp_size=tp_size,
            bs=bs,
            replicas=replicas,
            prompt=prompt,
            gen=gen,
            out_dir=args.out_dir,
        )
