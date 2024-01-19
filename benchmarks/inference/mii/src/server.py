# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import subprocess
import time

import mii
from deepspeed.inference import RaggedInferenceEngineConfig, DeepSpeedTPConfig
from deepspeed.inference.v2.ragged import DSStateManagerConfig

from .utils import parse_args, SERVER_PARAMS


def start_server(args):
    vllm = args.vllm
    model = args.model
    deployment_name = args.deployment_name
    tp_size = args.tp_size
    num_replicas = args.num_replicas
    max_ragged_batch_size = args.max_ragged_batch_size

    if vllm:
        start_vllm_server(model=model, tp_size=tp_size)
    else:
        start_mii_server(
            model=model,
            deployment_name=deployment_name,
            tp_size=tp_size,
            num_replicas=num_replicas,
            max_ragged_batch_size=max_ragged_batch_size,
        )


def start_vllm_server(model: str, tp_size: int) -> None:
    vllm_cmd = (
        "python",
        "-m",
        "vllm.entrypoints.api_server",
        "--host",
        "127.0.0.1",
        "--port",
        "26500",
        "--tensor-parallel-size",
        str(tp_size),
        "--model",
        model,
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
            stop_vllm_server()
            raise RuntimeError(f"Error starting VLLM server: {line}")
        if time.time() - start_time > timeout_after:
            p.terminate()
            stop_vllm_server()
            raise TimeoutError("Timed out waiting for VLLM server to start")
        time.sleep(0.01)


def start_mii_server(
    model, deployment_name, tp_size, num_replicas, max_ragged_batch_size
):
    tp_config = DeepSpeedTPConfig(tp_size=tp_size)
    mgr_config = DSStateManagerConfig(
        max_ragged_batch_size=max_ragged_batch_size,
        max_ragged_sequence_count=max_ragged_batch_size,
    )
    inference_config = RaggedInferenceEngineConfig(
        tensor_parallel=tp_config, state_manager=mgr_config
    )

    mii.serve(
        model,
        deployment_name=deployment_name,
        tensor_parallel=tp_size,
        inference_engine_config=inference_config,
        replica_num=num_replicas,
    )


def stop_server(args):
    vllm = args.vllm
    deployment_name = args.deployment_name

    if vllm:
        stop_vllm_server()
    else:
        stop_mii_server(deployment_name)


def stop_vllm_server():
    vllm_cmd = ("pkill", "-f", "vllm.entrypoints.api_server")
    p = subprocess.Popen(vllm_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()


def stop_mii_server(deployment_name):
    mii.client(deployment_name).terminate_server()


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
