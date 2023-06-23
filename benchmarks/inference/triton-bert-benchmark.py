# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import time
import deepspeed
import argparse
from transformers import pipeline
from deepspeed.accelerator import get_accelerator

def print_latency(latency_set, title, warmup=2, outliers=2):
    # trim warmup queries
    count = len(latency_set) - warmup - outliers
    assert count > 0 

    latency_set = latency_set[warmup:]
    avg = None
    if count > 0:
        latency_set.sort()
        latency_set = latency_set[:-outliers]
        n50 = (count - 1) * 0.5 + 1
        n90 = (count - 1) * 0.9 + 1
        n95 = (count - 1) * 0.95 + 1
        n99 = (count - 1) * 0.99 + 1
        n999 = (count - 1) * 0.999 + 1
        print(f"print_latency: {latency_set}, n90={n90}")

        avg = sum(latency_set) / count
        p50 = latency_set[int(n50) - 1]
        p90 = latency_set[int(n90) - 1]
        p95 = latency_set[int(n95) - 1]
        p99 = latency_set[int(n99) - 1]
        p999 = latency_set[int(n999) - 1]

        print(f"====== latency stats {title} ======")
        print("\tAvg Latency: {0:8.2f} ms".format(avg ))
        print("\tP50 Latency: {0:8.2f} ms".format(p50 ))
        print("\tP90 Latency: {0:8.2f} ms".format(p90 ))
        print("\tP95 Latency: {0:8.2f} ms".format(p95 ))
        print("\tP99 Latency: {0:8.2f} ms".format(p99 ))
        print("\t999 Latency: {0:8.2f} ms".format(p999 ))

    return avg, p90, p99

def get_dummy_query(seq_len, mask, task="fill-mask"):
    assert seq_len > 3 # +2 for special tokens such as CLS or SEP tokens (101 and 102 respectively) and +1 for mask
    query = f"{mask}" if task == "fill-mask" else f"end"
    for i in range(seq_len-3):
        query = "a " + query
    #print(query)
    return query

def run_benchmark(args, pipe, query=None, seq_len=None, use_triton=False, ds_run=True, task="fill-mask"):
    print(f"run_benchmark: {args}, {pipe}, {query}, {seq_len}, {use_triton}, {ds_run}, {task}")
    if args.dtype.lower() == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    if pipe is None:
        pipe = pipeline(task, model=args.model, framework="pt", device=args.local_rank)
        if dtype == torch.half:
            pipe.model.half()
        if ds_run:
            pipe.model = deepspeed.init_inference(pipe.model,
                                                    dtype=dtype,
                                                    mp_size=1,
                                                    replace_with_kernel_inject=args.kernel_inject,
                                                    replace_method='auto',
                                                    enable_cuda_graph=args.graphs,
                                                    use_triton=use_triton,
                                                    triton_autotune=True)
    else:
        if ds_run:
            pipe.model.cuda_graph_created = False

    if query is None:
        mask = pipe.tokenizer.mask_token
        if seq_len is None:
            query = f"Hello I am a {mask} model"
        else:
            query = get_dummy_query(seq_len, mask, task)

    profiling = False
    if profiling:
        from torch.profiler import profile, ProfilerActivity
        pipe(query)
        torch.cuda.synchronize() #ignore-cuda

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True, with_modules=True, profile_memory=True, with_stack=True) as prof:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
        # with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
            pipe(query)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
        print(prof.key_averages().table(row_limit=10))
        prof.export_chrome_trace(f"triton2-bert-benchmark-{seq_len}.json")
        torch.cuda.synchronize() #ignore-cuda

    if ds_run:
        pipe.model.profile_model_time()

    responses = []
    times = []
    mtimes = []
    for i in range(args.trials):
        get_accelerator().synchronize() #ignore-cuda
        start = time.time()
        r = pipe(query)
        get_accelerator().synchronize() #ignore-cuda
        end = time.time()
        e2e_time = (end - start)*1e3  # s to ms
        times.append(e2e_time)

        responses.append(r)

        mtime = pipe.model.model_times()
        mtimes += mtime

        print(f"trial{i}: e2e latency = {e2e_time}, model latency = {mtime}")

    _, e2e_latency, _ = print_latency(times, "e2e latency")
    _, model_latency, _ = print_latency(mtimes, "model latency")

    return e2e_latency, model_latency


def plot_lines(ys, x, ynames, ylabel='tops', title='', filename='lines.png'):
    import matplotlib.pyplot as plt
    for i, y in enumerate(ys):
        plt.plot(x, y, label=ynames[i])

    plt.legend()
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


# ex) deepspeed --num_gpus 1 triton-bert-benchmark.py --model bert-base-cased --dtype fp16 --kernel-inject --deepspeed --graphs --triton
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, help="hf model name")
    parser.add_argument("--dtype", type=str, default="fp16", help="fp16 or fp32")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--trials", type=int, default=8, help="number of trials")
    parser.add_argument("--kernel-inject", action="store_true", help="inject kernels on")
    parser.add_argument("--graphs", action="store_true", help="CUDA Graphs on")
    parser.add_argument("--triton", action="store_true", help="triton kernels on")
    parser.add_argument("--deepspeed", action="store_true", help="use deepspeed inference")
    parser.add_argument("--task", type=str, default="fill-mask", help="fill-mask or token-classification")
    args = parser.parse_args()

    deepspeed.init_distributed("nccl")
    print(args.model, args.dtype)
    print(args)

    pipe = pipeline(args.task, model=args.model, framework="pt", device=args.local_rank)
    if args.dtype.lower() == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    if dtype == torch.half:
        pipe.model.half()
    if args.deepspeed:
        pipe.model = deepspeed.init_inference(pipe.model,
                                                dtype=dtype,
                                                mp_size=1,
                                                replace_with_kernel_inject=args.kernel_inject,
                                                replace_method='auto',
                                                enable_cuda_graph=args.graphs,
                                                use_triton=args.triton,
                                                triton_autotune=True,
                                                max_out_tokens=pipe.tokenizer.model_max_length)
        pipe.model.profile_model_time()

    seq_lens =  [i for i in range(8,513,1)]
    seq_lens.sort()
    e2e_times = []
    model_times = []
    for seq_len in seq_lens:
        e2e_latency, model_latency = run_benchmark(args, pipe=pipe, query=None, seq_len=seq_len, use_triton=args.triton, ds_run=args.deepspeed, task=args.task)
        e2e_times.append(e2e_latency)
        model_times.append(model_latency)
    #plot_lines([e2e_times, model_times], seq_lens, ["e2e latency", "model_latency"], ylabel='avg ms', filename='triton-bert-bench.png')

    print("sequence length, e2e latency, model_latency")
    for i, t in enumerate(e2e_times):
        print(f"{seq_lens[i]}, {t}, {model_times[i]}")

    import numpy as np
    avg_latency = []
    start=0
    step=1
    steady_bin_size=64
    init_bin_size=(steady_bin_size-seq_lens[0])
    print("min-seq-len, max-seq-len, avg-latency(ms)")
    bins = range(int(np.ceil(len(seq_lens)/steady_bin_size)))
    for i in bins:
        if i == 0:
            bin_size = init_bin_size
            sidx = start 
            eidx = start + (i+1)*bin_size 
        else:
            bin_size = steady_bin_size
            sidx = eidx
            eidx = sidx + bin_size + (1 if i == len(bins) - 1 else 0)
        l = model_times[sidx:eidx]
        idx = seq_lens[sidx:eidx]
        print(f"{idx[0]}, {idx[-1]}, {np.mean(l)}")
    

