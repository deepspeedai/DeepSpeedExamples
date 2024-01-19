# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .postprocess_results import read_json, get_tokenizer

RAGGED_BATCH_SIZE = 768
SLA_PROMPT_TOKENS_PER_SEC = 512
SLA_GEN_TOKENS_PER_SEC = [1, 2, 3, 4, 6, 8]
EMA_SPAN = 16

tp_sizes_all = {"7b": [1], "70b": [4, 8]}

tp_sizes_test = {"7b": [1]}

prompt_gen_pairs_all = [
    (1200, 60),
    (1200, 128),
    (2600, 60),
    (2600, 128),
]

prompt_gen_pairs_test = [(2600, 60)]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--no_vllm", action="store_true")
    parser.add_argument("--log_dir", type=Path, default=".")
    parser.add_argument("--out_dir", type=Path, default="charts/goodtput")
    args = parser.parse_args()
    return args


def check_token_latency_step(response_details, token_index):
    P50_token_latency = np.percentile(
        [
            r.token_gen_time[token_index]
            for r in response_details
            if len(r.token_gen_time) > token_index
        ],
        50,
    )
    P90_token_latency = np.percentile(
        [
            r.token_gen_time[token_index]
            for r in response_details
            if len(r.token_gen_time) > token_index
        ],
        90,
    )
    P99_token_latency = np.percentile(
        [
            r.token_gen_time[token_index]
            for r in response_details
            if len(r.token_gen_time) > token_index
        ],
        99,
    )

    return P50_token_latency, P90_token_latency, P99_token_latency


def validate_token_cum_latency_SLA(response_detail, sla_token_gen):
    cumsum_latencies = np.cumsum(np.array(response_detail.token_gen_time[1:]))
    return all(
        [
            cumsum_latencies[i] <= (1 / sla_token_gen) * (i + 1)
            for i in range(len(cumsum_latencies))
        ]
    )


def validate_token_ema_latency_SLA(response_detail, sla_token_gen, ema_span):
    ema_latency = (
        pd.Series(response_detail.token_gen_time[1:])
        .ewm(span=ema_span)
        .mean()
        .values.tolist()
    )
    return all([t < 1.0 / sla_token_gen for t in ema_latency])


def validate_prompt_latency_SLA(response_detail, sla_token_gen, f):
    tokenizer = get_tokenizer()
    prompt_length = len(tokenizer.tokenize(response_detail.prompt))
    prompt_latency_SLA = prompt_length / SLA_PROMPT_TOKENS_PER_SEC
    if prompt_latency_SLA < response_detail.token_gen_time[0]:
        return False

    if len(response_detail.token_gen_time) == 1:
        return True

    return f[0](response_detail, sla_token_gen, *f[1])


def calc_throughput(response_details):
    start_time = min([r.start_time for r in response_details])
    end_time = max([r.end_time for r in response_details])
    return len(response_details) / (end_time - start_time)


def extract_values(file_pattern, sla_token_gen, validate_func):
    files = glob.glob(file_pattern)
    print(f"Found {len(files)} files")
    goodputs = {}
    good_ratios = {}
    for f in files:
        prof_args, response_details = read_json(f)
        client_num = prof_args["client_num"]
        num_req_ok = len(
            [
                r
                for r in response_details
                if validate_prompt_latency_SLA(r, sla_token_gen, validate_func)
            ]
        )
        goodputs[client_num] = calc_throughput(response_details) * (
            num_req_ok / len(response_details)
        )
        good_ratios[client_num] = num_req_ok / len(response_details)

    return goodputs, good_ratios


def display_results(model_size, tp, bs, sla_token_gen, prompt, gen, log_dir, out_dir):
    if not log_dir.exists():
        print(f"Log directory {log_dir} does not exist")
        return

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"model: {model_size} Prompt: {prompt}, Generation: {gen}, TP: {tp} sla_token_gen: {sla_token_gen}"
    )

    mii_file_pattern = f"{log_dir}/logs.llama2-{model_size}-tp{tp}-b{bs}/llama2-{model_size}-tp{tp}-b{bs}_c*_p{prompt}_g{gen}.json"
    if not args.no_vllm:
        vllm_file_pattern = f"{log_dir}/logs.vllm-llama2-{model_size}-tp{tp}/vllm-llama2-{model_size}-tp{tp}_c*_p{prompt}_g{gen}.json"

    validate_funcs = [
        (validate_token_cum_latency_SLA, (), "cum"),
        (validate_token_ema_latency_SLA, (EMA_SPAN,), f"ema{EMA_SPAN}"),
    ]

    for f in validate_funcs:

        mii_goodputs, mii_good_ratios = extract_values(
            mii_file_pattern, sla_token_gen, f
        )
        client_num_list = sorted(list(mii_goodputs.keys()))
        mii_goodputs_list = [mii_goodputs[client_num] for client_num in client_num_list]

        if not args.no_vllm:
            vllm_goodputs, vllm_good_ratios = extract_values(
                vllm_file_pattern, sla_token_gen, f
            )
            vllm_goodputs_list = [
                vllm_goodputs[client_num] for client_num in client_num_list
            ]

        # print(f"MII {mii_goodputs_list} ratio={mii_good_ratios}")
        # print(f"vLLM {vllm_goodputs_list} ratio={vllm_good_ratios}")

        # Plotting the scatter plot
        plt.figure(figsize=(7, 4))
        plt.scatter(
            client_num_list,
            mii_goodputs_list,
            label=f"DeepSpeed-FastGen",
            marker="o",
            color="blue",
        )
        if not args.no_vllm:
            plt.scatter(
                client_num_list,
                vllm_goodputs_list,
                label=f"vLLM",
                marker="x",
                color="orange",
            )

        fit_x_list = np.arange(min(client_num_list), max(client_num_list), 0.1)
        mii_fit_model = np.polyfit(client_num_list, mii_goodputs_list, 4)
        mii_model_fn = np.poly1d(mii_fit_model)
        plt.plot(
            fit_x_list,
            mii_model_fn(fit_x_list),
            color="blue",
            alpha=0.5,
            linestyle="--",
        )

        if not args.no_vllm:
            vllm_fit_model = np.polyfit(client_num_list, vllm_goodputs_list, 4)
            vllm_model_fn = np.poly1d(vllm_fit_model)
            plt.plot(
                fit_x_list,
                vllm_model_fn(fit_x_list),
                color="orange",
                alpha=0.5,
                linestyle="--",
            )

        title = (
            f"Effective throughput (SLA prompt: {SLA_PROMPT_TOKENS_PER_SEC} tokens/s, generation: {sla_token_gen} tokens/s)\n"
            + f"Llama 2 {model_size.upper()} Prompt: {prompt}, Generation: {gen}, TP: {tp}"
        )
        plt.title(title, fontsize=10)
        plt.xlabel("Number of clients", fontsize=10)
        plt.ylabel("Effective throughput (queries/s)", fontsize=10)
        # plt.rcParams['figure.subplot.bottom'] = 0.30
        plt.ylim(bottom=-0.05)
        plt.legend()
        plt.grid(True)
        # plt.show()
        out_file = (
            out_dir
            / f"goodput_llama{model_size}_SLAp{SLA_PROMPT_TOKENS_PER_SEC}g{sla_token_gen}_tp{tp}_b{bs}_p{prompt}g{gen}_{f[2]}.png"
        )
        plt.savefig(out_file)
        plt.clf()
        print(f"Saved {out_file}")


if __name__ == "__main__":
    raise NotImplementedError("This script is not up to date")
    args = get_args()

    if args.test:
        tp_sizes = tp_sizes_test
        prompt_gen_pairs = prompt_gen_pairs_test
    else:
        tp_sizes = tp_sizes_all
        prompt_gen_pairs = prompt_gen_pairs_all

    for model_size, tps in tp_sizes.items():
        for tp in tps:
            for prompt, gen in prompt_gen_pairs:
                for sla_token_gen in SLA_GEN_TOKENS_PER_SEC:
                    display_results(
                        model_size,
                        tp,
                        RAGGED_BATCH_SIZE,
                        sla_token_gen,
                        prompt,
                        gen,
                        args.log_dir,
                        args.out_dir,
                    )
