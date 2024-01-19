# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import asyncio
import json
import multiprocessing
import os
import queue
import random
import requests
import threading
import time
from typing import List, Iterable

import numpy as np
from transformers import AutoTokenizer

from .postprocess_results import ResponseDetails
from .random_query_generator import RandomQueryGenerator
from .sample_input import all_text
from .utils import parse_args, print_summary, get_args_product, CLIENT_PARAMS


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
            input_tokens, max_new_tokens=max_new_tokens, streaming_fn=callback
        )
    else:
        result = client.generate(input_tokens, max_new_tokens=max_new_tokens)
        output_tokens = result[0].generated_text

    return ResponseDetails(
        generated_tokens=output_tokens,
        prompt=input_tokens,
        start_time=start_time,
        end_time=time.time(),
        model_time=0,
        token_gen_time=token_gen_time,
    )


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
        LINE_UP = "\033[1A"
        LINE_CLEAR = "\x1b[2K"
        for _ in range(n):
            print(LINE_UP, end=LINE_CLEAR, flush=True)

    def get_streaming_response(
        response: requests.Response, time_last_token
    ) -> Iterable[List[str]]:
        for chunk in response.iter_lines(
            chunk_size=8192, decode_unicode=False, delimiter=b"\0"
        ):
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
            token_gen_time=token_gen_time,
        )
    else:
        output = get_response(response)
        raise NotImplementedError("Not implemented for non-streaming")


def _run_parallel(
    deployment_name,
    warmup,
    barrier,
    query_queue,
    result_queue,
    num_clients,
    stream,
    vllm,
):
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

    time.sleep(random.uniform(0, num_clients) * 0.01)
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


def run_client(args):
    """
    Run MII client for benchmarking. The scenario is a bit complicated:
    1. The main process puts `num_requests` queries into the input queue
    2. Each client runs `warmup` iterations () taking the queries from the input queue
    3. --- barrier ---
    4. The main process marks the start time
    5a. All clients send `num_requests' query in total and put the results into the result queue
    5b. The main process takes the results from the result queue (in parallel with 5a)
    6. The main process marks the end time after receiving `num_requests' results
    """

    # Unpack arguments
    model = args.model
    deployment_name = args.deployment_name
    mean_prompt_length = args.mean_prompt_length
    mean_max_new_tokens = args.mean_max_new_tokens
    num_clients = args.num_clients
    num_requests = args.num_requests
    warmup = args.warmup
    max_prompt_length = args.max_prompt_length
    prompt_length_var = args.prompt_length_var
    max_new_tokens_var = args.max_new_tokens_var
    stream = args.stream
    vllm = args.vllm
    use_thread = args.use_thread

    if use_thread:
        runnable_cls = threading.Thread
        barrier_cls = threading.Barrier
        queue_cls = queue.Queue
    else:
        runnable_cls = multiprocessing.Process
        barrier_cls = multiprocessing.Barrier
        queue_cls = multiprocessing.Queue

    barrier = barrier_cls(num_clients + 1)
    query_queue = queue_cls()
    result_queue = queue_cls()

    processes = [
        runnable_cls(
            target=_run_parallel,
            args=(
                deployment_name,
                warmup,
                barrier,
                query_queue,
                result_queue,
                num_clients,
                stream,
                vllm,
            ),
        )
        for i in range(num_clients)
    ]
    for p in processes:
        p.start()

    tokenizer = AutoTokenizer.from_pretrained(model)
    query_generator = RandomQueryGenerator(all_text, tokenizer, seed=42)
    request_text = query_generator.get_random_request_text(
        mean_prompt_length,
        mean_prompt_length * prompt_length_var,
        max_prompt_length,
        num_requests + warmup * num_clients,
    )

    for t in request_text:
        req_max_new_tokens = int(
            np.random.normal(
                mean_max_new_tokens, max_new_tokens_var * mean_max_new_tokens
            )
        )
        query_queue.put((t, req_max_new_tokens))

    # Tokenizers must be initialized after fork.
    # So we need to fork before putting inputs to the queue.
    # We need this barrier to stop child processse from taking inputs before the main process puts them
    barrier.wait()
    # This barrier is to make sure that all clients have finished warmup
    barrier.wait()

    response_details = []
    while len(response_details) < num_requests:
        res = result_queue.get()
        # vLLM returns concatinated tokens
        if vllm:
            all_tokens = tokenizer.tokenize(res.generated_tokens)
            res.generated_tokens = all_tokens[len(tokenizer.tokenize(res.prompt)) :]
        response_details.append(res)

    return response_details


if __name__ == "__main__":
    args = parse_args(client_args=True)

    for client_args in get_args_product(args, which=CLIENT_PARAMS):
        response_details = run_client(client_args)

        print_summary(client_args, response_details)
