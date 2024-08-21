# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from postprocess_results import read_json, get_tokenizer, get_result_sets


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=["fastgen", "vllm", "openai"], default=["fastgen", "vllm"], \
                        nargs="+", help="Specify the backends to generate plots for")
    parser.add_argument("--log_dir", type=Path, default="./results")
    parser.add_argument("--model", type=str)
    parser.add_argument("--out_dir", type=Path, default="./plots/goodtput")
    parser.add_argument("--sla_prompt_tokens_per_sec", type=int, default=512, help="SLA prompt tokens per second")
    parser.add_argument("--sla_gen_tokens_per_sec", type=int, default=[1, 2, 3, 4, 6, 8], nargs="+", help="SLA generation tokens/s targets")
    parser.add_argument("--ema_span", type=int, default=16, help="EMA span")
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


def validate_prompt_latency_SLA(response_detail, sla_token_gen, f, sla_prompt_tokens_per_sec ):
    tokenizer = get_tokenizer(args.model)
    prompt_length = len(tokenizer.tokenize(response_detail.prompt))
    prompt_latency_SLA = prompt_length / sla_prompt_tokens_per_sec
    if prompt_latency_SLA < response_detail.token_gen_time[0]:
        return False

    if len(response_detail.token_gen_time) == 1:
        return True

    return f[0](response_detail, sla_token_gen, *f[1])


def calc_throughput(response_details):
    start_time = min([r.start_time for r in response_details])
    end_time = max([r.end_time for r in response_details])
    return len(response_details) / (end_time - start_time)


def extract_values(file_pattern, sla_token_gen, validate_func, sla_prompt_tokens_per_sec):
    files = glob.glob(file_pattern)
    print(f"Found {len(files)} files")
    goodputs = {}
    good_ratios = {}
    for f in files:
        prof_args, response_details = read_json(f)
        client_num = prof_args["num_clients"]
        num_req_ok = len(
            [
                r
                for r in response_details
                if validate_prompt_latency_SLA(r, sla_token_gen, validate_func, sla_prompt_tokens_per_sec)
            ]
        )
        goodputs[client_num] = calc_throughput(response_details) * (
            num_req_ok / len(response_details)
        )
        good_ratios[client_num] = num_req_ok / len(response_details)

    return goodputs, good_ratios


def output_charts(args, model, tp_size, bs, replicas, sla_token_gen, prompt, gen, log_dir, out_dir):
    if not log_dir.exists():
        print(f"Log directory {log_dir} does not exist")
        return

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Model: {model} Prompt: {prompt}, Generation: {gen}, TP: {tp_size} sla_token_gen: {sla_token_gen}"
    )

    result_file_pattern = f"{model}-tp{tp_size}-bs{bs}-replicas{replicas}-prompt{prompt}-gen{gen}-clients*.json"

    validate_funcs = [
        (validate_token_cum_latency_SLA, (), "cum"),
        (validate_token_ema_latency_SLA, (args.ema_span,), f"ema{args.ema_span}"),
    ]

    plt_cfg = {'vllm': {'label': 'vLLM', 'marker': 'x', 'color': 'orange'},\
               'fastgen': {'label': 'DeepSpeed-FastGen', 'marker': 'o', 'color': 'blue'}, \
               'openai': {'label': 'openai-API', 'marker': '+', 'color': 'red'}
              }

    for f in validate_funcs:
        plt.figure()

        for backend in args.backend:
            file_pattern = f"{log_dir}/{backend}/{result_file_pattern}"
            goodputs, good_ratios = extract_values(
                file_pattern, sla_token_gen, f, args.sla_prompt_tokens_per_sec
            )
            client_num_list = sorted(list(goodputs.keys()))
            goodputs_list = [goodputs[client_num] for client_num in client_num_list]

            # Plotting the scatter plot
            plt.scatter(
                client_num_list,
                goodputs_list,
                label=plt_cfg[backend]['label'],
                marker=plt_cfg[backend]['marker'],
                color=plt_cfg[backend]['color'],
            )

            fit_x_list = np.arange(min(client_num_list), max(client_num_list), 0.1)
            fit_model = np.polyfit(client_num_list, goodputs_list, 4)
            model_fn = np.poly1d(fit_model)
            plt.plot(
                fit_x_list,
                model_fn(fit_x_list),
                alpha=0.5,
                linestyle="--",
                color=plt_cfg[backend]['color'],
            )

        title = (
            f"Effective throughput (SLA prompt: {args.sla_prompt_tokens_per_sec} tokens/s, generation: {sla_token_gen} tokens/s)\n"
            + f"Model: {model} Prompt: {prompt}, Generation: {gen}, TP: {tp_size}"
        )
        plt.title(title, fontsize=10)
        plt.xlabel("Number of clients", fontsize=10)
        plt.ylabel("Effective throughput (queries/s)", fontsize=10)
        plt.ylim(bottom=-0.05)
        plt.legend()
        plt.grid(True)
        out_file = (
            out_dir
            / f"{model}_SLAp{args.sla_prompt_tokens_per_sec}g{sla_token_gen}_tp{tp_size}_b{bs}_p{prompt}g{gen}_{f[2]}.png"
        )
        plt.savefig(out_file)
        plt.clf()
        print(f"Saved {out_file}")


if __name__ == "__main__":
    args = get_args()

    assert "aml" not in args.backend, "Effective throughput analysis is not supported for AML."

    result_params = get_result_sets(args)

    for model, tp_size, bs, replicas, prompt, gen in result_params:
        for sla_token_gen in args.sla_gen_tokens_per_sec:
             output_charts(
                args=args,
                model=model,
                tp_size=tp_size,
                bs=bs,
                replicas=replicas,
                sla_token_gen=sla_token_gen,
                prompt=prompt,
                gen=gen,
                log_dir=args.log_dir,
                out_dir=args.out_dir,
            )
