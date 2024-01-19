# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import itertools

from .postprocess_results import read_json, get_token_latency

bs = 768
SKIP_HEAD_TOKEN_NUM = 2
SKIP_REQUEST_NUM = 100

tp_sizes = {
    "70b": [4],
}

prompt_gen_pairs = [
    (2600, 128),
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=Path, default=".")
    parser.add_argument(
        "--out_dir", type=Path, default="charts/percentile_token_latency"
    )
    args = parser.parse_args()
    return args


def extract_values(file_pattern):
    files = glob.glob(file_pattern)

    latencies = {}
    for f in files:
        prof_args, response_details = read_json(f)
        client_num = prof_args["client_num"]

        response_details.sort(key=lambda r: r.start_time)
        response_details = response_details[SKIP_REQUEST_NUM:-SKIP_REQUEST_NUM]
        token_latencies = [
            r.token_gen_time[SKIP_HEAD_TOKEN_NUM:-1] for r in response_details
        ]

        flat_latency_list = list(itertools.chain(*token_latencies))
        latencies[client_num] = flat_latency_list
    return latencies


def output_charts(model_size, tp, bs, prompt, gen, log_dir, out_dir):
    if not log_dir.exists():
        print(f"Log directory {log_dir} does not exist")
        return

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    mii_file_pattern = f"{log_dir}/logs.llama2-{model_size}-tp{tp}-b{bs}/llama2-{model_size}-tp{tp}-b{bs}_c*_p{prompt}_g{gen}.json"
    vllm_file_pattern = f"{log_dir}/logs.vllm-llama2-{model_size}-tp{tp}/vllm-llama2-{model_size}-tp{tp}_c*_p{prompt}_g{gen}.json"

    mii_latencies = extract_values(mii_file_pattern)
    vllm_latencies = extract_values(vllm_file_pattern)
    client_num_list = sorted(list(mii_latencies.keys()))

    for client_num in client_num_list:
        plt.figure(figsize=(6, 4))

        percentile = 95

        P50_vllm_val = np.percentile(vllm_latencies[client_num], 50)
        P50_mii_val = np.percentile(mii_latencies[client_num], 50)
        P90_vllm_val = np.percentile(vllm_latencies[client_num], 90)
        P90_mii_val = np.percentile(mii_latencies[client_num], 90)
        P95_vllm_val = np.percentile(vllm_latencies[client_num], 95)
        P95_mii_val = np.percentile(mii_latencies[client_num], 95)

        # print(f"P50_vllm_val={P50_vllm_val}")
        # print(f"P50_mii_val={P50_mii_val}")
        # print(f"P90_vllm_val={P90_vllm_val}")
        # print(f"P90_mii_val={P90_mii_val}")
        # print(f"P95_vllm_val={P95_vllm_val}")
        # print(f"P95_mii_val={P95_mii_val}")

        out_file = (
            out_dir
            / f"p{percentile}_token_latency_llama{model_size}_c{client_num}_tp{tp}_p{prompt}g{gen}.png"
        )

        x1 = [1, 2, 3]
        y1 = [P50_vllm_val, P90_vllm_val, P95_vllm_val]

        x2 = [1.3, 2.3, 3.3]
        y2 = [P50_mii_val, P90_mii_val, P95_mii_val]

        label_x = ["P50", "P90", "P95"]

        plt.bar(x1, y1, width=0.3, label="vLLM", align="center", color="orange")
        plt.bar(
            x2, y2, width=0.3, label="DeepSpeed-FastGen", align="center", color="blue"
        )
        plt.ylabel("Latency", fontsize=14)
        plt.legend(loc=2)

        plt.xticks([1.15, 2.15, 3.15], label_x)

        plt.savefig(out_file)
        print(f"Saved {out_file}")


if __name__ == "__main__":
    raise NotImplementedError("This script is not up to date")
    args = get_args()

    for model_size, tps in tp_sizes.items():
        for tp in tps:
            for prompt, gen in prompt_gen_pairs:
                output_charts(
                    model_size, tp, bs, prompt, gen, args.log_dir, args.out_dir
                )
