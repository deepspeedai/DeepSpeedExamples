# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
import subprocess
import time


try:
    from .utils import parse_args, SERVER_PARAMS
except ImportError:
    from utils import parse_args, SERVER_PARAMS


def start_server(args: argparse.Namespace) -> None:
    start_server_fns = {
        "fastgen": start_fastgen_server,
        "vllm": start_vllm_server,
        "aml": start_aml_server,
        "openai": start_openai_server,
    }
    start_fn = start_server_fns[args.backend]
    start_fn(args)


def start_vllm_server(args: argparse.Namespace) -> None:
    vllm_cmd = (
        "python",
        "-m",
        "vllm.entrypoints.api_server",
        "--host",
        "127.0.0.1",
        "--port",
        "26500",
        "--tensor-parallel-size",
        str(args.tp_size),
        "--model",
        args.model,
    )
    p = subprocess.Popen(
        vllm_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, close_fds=True
    )
    start_time = time.time()
    timeout_after = 60 * 5  # 5 minutes
    while True:
        line = p.stderr.readline().decode("utf-8")
        if "Application startup complete" in line:
            break
        if "error" in line.lower():
            p.terminate()
            stop_vllm_server(args)
            raise RuntimeError(f"Error starting VLLM server: {line}")
        if time.time() - start_time > timeout_after:
            p.terminate()
            stop_vllm_server(args)
            raise TimeoutError("Timed out waiting for VLLM server to start")
        time.sleep(0.01)


def start_fastgen_server(args: argparse.Namespace) -> None:
    import mii
    from deepspeed.inference import RaggedInferenceEngineConfig, DeepSpeedTPConfig
    from deepspeed.inference.v2.ragged import DSStateManagerConfig

    tp_config = DeepSpeedTPConfig(tp_size=args.tp_size)
    mgr_config = DSStateManagerConfig(
        max_ragged_batch_size=args.max_ragged_batch_size,
        max_ragged_sequence_count=args.max_ragged_batch_size,
    )
    inference_config = RaggedInferenceEngineConfig(
        tensor_parallel=tp_config, state_manager=mgr_config
    )
    if args.fp6:
        quantization_mode = 'wf6af16'
    else:
        quantization_mode = None
    mii.serve(
        args.model,
        deployment_name=args.deployment_name,
        tensor_parallel=args.tp_size,
        inference_engine_config=inference_config,
        replica_num=args.num_replicas,
        quantization_mode=quantization_mode
    )


def start_aml_server(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "AML server start not implemented. Please use Azure Portal to start the server."
    )

def start_openai_server(args: argparse.Namespace) -> None:
    # openai api has no command to stop server
    pass

def stop_server(args: argparse.Namespace) -> None:
    stop_server_fns = {
        "fastgen": stop_fastgen_server,
        "vllm": stop_vllm_server,
        "aml": stop_aml_server,
        "openai": stop_openai_server,
    }
    stop_fn = stop_server_fns[args.backend]
    stop_fn(args)


def stop_vllm_server(args: argparse.Namespace) -> None:
    vllm_cmd = ("pkill", "-f", "vllm.entrypoints.api_server")
    p = subprocess.Popen(vllm_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()


def stop_fastgen_server(args: argparse.Namespace) -> None:
    import mii

    mii.client(args.deployment_name).terminate_server()


def stop_aml_server(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "AML server stop not implemented. Please use Azure Portal to stop the server."
    )

def stop_openai_server(args: argparse.Namespace) -> None:
    # openai api has no command to stop server
    pass

if __name__ == "__main__":
    args = parse_args(server_args=True)

    if args.cmd == "start":
        start_server(args)
    elif args.cmd == "stop":
        stop_server(args)
    elif args.cmd == "restart":
        stop_server(args)
        start_server(args)
    else:
        raise ValueError(f"Invalid command {args.cmd}")
