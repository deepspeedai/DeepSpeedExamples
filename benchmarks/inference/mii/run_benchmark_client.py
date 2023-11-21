import os
import time
import random
import argparse
import queue
import multiprocessing
import threading
from statistics import mean
from dataclasses import dataclass, asdict
from typing import List, Iterable
from pathlib import Path
from datetime import datetime
import numpy as np

from transformers import AutoTokenizer
from random_query_generator import RandomQueryGenerator
from sample_input import all_text
import time
import json
import asyncio
import requests

from postprocess_results import get_summary, ResponseDetails

MAX_PROMPT_LENGTH = 4000
PROMPT_LENGTH_VAR = 0.3
MAX_NEW_TOKENS_VAR = 0.3

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark MII services")
    parser.add_argument("-k",
                        "--max_new_tokens",
                        type=int,
                        default=60,
                        help="min and max num tokens argument for huggingface")
    parser.add_argument("-d",
                        "--deployment_name",
                        type=str,
                        default="benchmark_deployment")
    parser.add_argument("-n",
                        "--num_queries",
                        type=int,
                        help="number of queries to run",
                        default=10)
    parser.add_argument("-w",
                        "--warmup",
                        type=int,
                        help="number of queries for warming up",
                        default=1)
    parser.add_argument("-c",
                        "--client_num",
                        type=int,
                        help="number of parallel client processes",
                        default=2)
    parser.add_argument("-l",
                        "--prompt_length",
                        type=int,
                        default=2600)
    parser.add_argument('--use_thread', action='store_true',
                        help='use thread to run parallel clients, otherwise use multiprocessing',
                        default=False)
    parser.add_argument('--stream', action='store_true', default=True)
    parser.add_argument('--vllm', action='store_true', default=False)
    parser.add_argument('-o', '--out_json_path', type=Path, default=None)

    args = parser.parse_args()
    return args


def call_mii(client, input_tokens, max_new_tokens, stream):
    output_tokens = []
    token_gen_time = []
    time_last_token = 0

    def callback(response):
        nonlocal time_last_token
        # print(f"Received: {response[0].generated_text} time_last_token={time_last_token}")
        output_tokens.append(response[0].generated_text)
        time_now = time.time()
        token_gen_time.append(time_now - time_last_token)
        time_last_token = time_now

    time_last_token = start_time = time.time()
    token_gen_time = []
    if stream:
        output_tokens = []
        client.generate(
            input_tokens, max_new_tokens=max_new_tokens,
            streaming_fn=callback)
    else:
        result = client.generate(
            input_tokens, max_new_tokens=max_new_tokens)
        output_tokens = result[0].generated_text

    return ResponseDetails(
        generated_tokens=output_tokens,
        prompt=input_tokens,
        start_time=start_time,
        end_time=time.time(),
        model_time=0,
        token_gen_time=token_gen_time)


def call_vllm(input_tokens, max_new_tokens, stream=True):
    api_url = "http://localhost:26500/generate"
    headers = {"User-Agent": "Benchmark Client"}
    pload = {
        "prompt": input_tokens,
        "n": 1,
        "use_beam_search": False,
        "temperature": 1.0,
        "top_p": 0.9,
        "max_tokens": max_new_tokens,
        "ignore_eos": False,
        "stream": stream,
    }
    def clear_line(n: int = 1) -> None:
        LINE_UP = '\033[1A'
        LINE_CLEAR = '\x1b[2K'
        for _ in range(n):
            print(LINE_UP, end=LINE_CLEAR, flush=True)

    def get_streaming_response(response: requests.Response, time_last_token) -> Iterable[List[str]]:
        for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False,
                                        delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode("utf-8"))
                output = data["text"][0]
                time_now = time.time()
                yield output, time_now - time_last_token
                time_last_token = time_now

    def get_response(response: requests.Response) -> List[str]:
        data = json.loads(response.content)
        output = data["text"]
        return output

    start_time = time.time()
    response = requests.post(api_url, headers=headers, json=pload, stream=stream)
    if stream:
        token_gen_time = []
        for h, t in get_streaming_response(response, start_time):
            output = h
            token_gen_time.append(t)

        return ResponseDetails(
            generated_tokens=output,
            prompt=input_tokens,
            start_time=start_time,
            end_time=time.time(),
            model_time=0,
            token_gen_time=token_gen_time)
    else:
        output = get_response(response)
        raise NotImplementedError("Not implemented for non-streaming")


def _run_parallel(deployment_name, warmup, barrier, query_queue, result_queue, client_num, stream, vllm):
    pid = os.getpid()
    session_id = f"test_session_p{pid}_t{threading.get_ident()}"

    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    if not vllm:
        import mii
        client = mii.client(deployment_name)

    barrier.wait()

    for _ in range(warmup):
        print(f"warmup queue size: {query_queue.qsize()} ({pid})", flush=True)
        input_tokens, req_max_new_tokens = query_queue.get(timeout=1.0)

        if vllm:
            call_vllm(input_tokens, req_max_new_tokens, stream)
        else:
            call_mii(client, input_tokens, req_max_new_tokens, stream)

    barrier.wait()

    time.sleep(random.uniform(0, client_num) * 0.01)
    try:
        while not query_queue.empty():
            print(f"queue size: {query_queue.qsize()} ({pid})", flush=True)
            input_tokens, req_max_new_tokens = query_queue.get(timeout=1.0)

            # Set max_new_tokens following normal distribution
            if vllm:
                r = call_vllm(input_tokens, req_max_new_tokens)
            else:
                r = call_mii(client, input_tokens, req_max_new_tokens, stream)

            result_queue.put(r)
    except queue.Empty:
        print(f"queue is empty ({pid})")

    print(f"Worker ({pid}) finished. session_id: {session_id}")


def run_client(client_num, deployment_name, prompt_length, max_new_tokens, num_queries, warmup, stream, vllm, use_thread=False):
    """
    Run MII client for benchmarking. The scenario is a bit complicated:
    1. The main process puts `num_queries` queries into the input queue
    2. Each client runs `warmup` iterations () taking the queries from the input queue
    3. --- barrier ---
    4. The main process marks the start time
    5a. All clients send `num_queries' query in total and put the results into the result queue
    5b. The main process takes the results from the result queue (in parallel with 5a)
    6. The main process marks the end time after receiving `num_queries' results
    """

    if use_thread:
        runnable_cls = threading.Thread
        barrier_cls = threading.Barrier
        queue_cls = queue.Queue
    else:
        runnable_cls = multiprocessing.Process
        barrier_cls = multiprocessing.Barrier
        queue_cls = multiprocessing.Queue

    barrier = barrier_cls(client_num + 1)
    query_queue = queue_cls()
    result_queue = queue_cls()

    processes = [runnable_cls(target=_run_parallel,
                              args=(deployment_name, warmup, barrier, query_queue, result_queue, client_num, stream, vllm))
                 for i in range(client_num)]
    for p in processes:
        p.start()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    query_generator = RandomQueryGenerator(all_text, tokenizer, seed=42)
    MAX_PROMPT_LENGTH = 4000
    request_text = query_generator.get_random_request_text(prompt_length, prompt_length*PROMPT_LENGTH_VAR, MAX_PROMPT_LENGTH, num_queries + warmup*client_num)

    for t in request_text:
        req_max_new_tokens = int(np.random.normal(max_new_tokens, MAX_NEW_TOKENS_VAR*max_new_tokens))
        query_queue.put((t, req_max_new_tokens))

    # Tokenizers must be initialized after fork.
    # So we need to fork before putting inputs to the queue.
    # We need this barrier to stop child processse from taking inputs before the main process puts them
    barrier.wait()
    # This barrier is to make sure that all clients have finished warmup
    barrier.wait()

    response_details = []
    while len(response_details) < num_queries:
        res = result_queue.get()
        # vLLM returns concatinated tokens
        if vllm:
            all_tokens = tokenizer.tokenize(res.generated_tokens)
            res.generated_tokens = all_tokens[len(tokenizer.tokenize(res.prompt)):]
        response_details.append(res)

    return response_details

if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.out_json_path is not None and not args.out_json_path.parent.exists():
        raise ValueError(f"Parent directory of {args.out_json_path}")

    response_details = run_client(args.client_num, args.deployment_name,
                            args.prompt_length,
                            args.max_new_tokens, args.num_queries, args.warmup,
                            args.stream, args.vllm, args.use_thread)

    args_dict = vars(args)
    ps = get_summary(args_dict, response_details)
    print(f"Deployment: {args.deployment_name} Clients: {args.client_num}, "
          + f"Prompt (mean): {args.prompt_length} tokens, "
          + f"Generation (mean): {args.max_new_tokens} tokens, "
          + f"Query throughput: {ps.throughput:.3f} queries/s, "
          + f"Token throughput (total): {ps.tokens_per_sec:.3f} tokens/s, "
          + f"Query latency: {ps.latency:.3f} s, "
          + f"Token generation latency: {ps.token_gen_latency:.3f} s/token, "
          + f"First token received: {ps.first_token_latency:.3f} s")

    if args.out_json_path is not None:
        with open(args.out_json_path, "w") as f:
            args_dict["out_json_path"] = str(args.out_json_path) # Path is not JSON serializable
            data = {"args": args_dict, "time": str(datetime.now()), "response_details": [asdict(r) for r in response_details]}
            json.dump(data, f, indent=2)
