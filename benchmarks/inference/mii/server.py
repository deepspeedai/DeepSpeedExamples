# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import mii
import argparse

from mii.constants import DeploymentType

from deepspeed.inference import RaggedInferenceEngineConfig, DeepSpeedTPConfig
from deepspeed.inference.v2.ragged import DSStateManagerConfig

def start_server(model_name,
                 deployment_name,
                 task,
                 tensor_parallel,
                 replica_num,
                 max_ragged_batch_size):
    tp_config = DeepSpeedTPConfig(tp_size=tensor_parallel)
    mgr_config = DSStateManagerConfig(max_ragged_batch_size=max_ragged_batch_size, max_ragged_sequence_count=max_ragged_batch_size)
    inference_config = RaggedInferenceEngineConfig(tensor_parallel=tp_config,
                                                   state_manager=mgr_config)

    mii.serve(
        model_name,
        deployment_name=deployment_name,
        tensor_parallel=tensor_parallel,
        task=task,
        inference_engine_config=inference_config,
        replica_num=replica_num
    )

def stop_server(deployment_name):
    mii.client(deployment_name).terminate_server()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        default="meta-llama/Llama-2-7b-hf",
                        help="Name of the model in the model_files to benchmark")
    parser.add_argument("-d",
                        "--deployment_name",
                        type=str,
                        default="benchmark_deployment")
    parser.add_argument("-t", "--task", type=str,
                        help="Task type. Currently only text-generation is supported",
                        default="text-generation")
    parser.add_argument("-m",
                        "--tensor_parallel",
                        type=int,
                        help="Degree of tensor (model) parallelism",
                        default=1)
    parser.add_argument("-b",
                        "--ragged_batch_size",
                        type=int,
                        help="Max batch size for ragged batching",
                        default=768)
    parser.add_argument("-r",
                        "--replica_num",
                        type=int,
                        help="Number of replicas for load balancing",
                        default=1)
    parser.add_argument("cmd", help="start, stop, or restart")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.cmd == "start":
        start_server(args.model_name,
                     args.deployment_name,
                     args.task,
                     args.tensor_parallel,
                     args.replica_num,
                     args.ragged_batch_size)
    elif args.cmd == "stop":
        print("running stop")
        stop_server(args.deployment_name)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")
