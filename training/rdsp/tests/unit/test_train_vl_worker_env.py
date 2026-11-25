# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""train_vl.worker_env: the driver's settings Ray gives every worker of the job, so actors
on a cluster started apart from the driver (`ray start` on each of two nodes) see them."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

import train_vl  # noqa: E402


def test_passes_what_actors_read_and_nothing_else():
    env = {
        "RDSP_PROFILE": "1",
        "RDSP_P2P_TIMEOUT_S": "600",
        "NCCL_SOCKET_IFNAME": "^lo,docker",
        "FI_PROVIDER": "efa",
        "HF_HOME": "/home/ubuntu/hf",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "DS_BUILD_OPS": "0",
        "CUDA_HOME": "/usr/local/cuda-13.0",
        "LD_LIBRARY_PATH": "/opt/amazon/ofi-nccl/lib",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "HOME": "/home/ubuntu",
        "PATH": "/usr/bin",
        "RAY_ADDRESS": "auto",
    }
    got = train_vl.worker_env(env)
    assert got == {
        k: env[k]
        for k in (
            "RDSP_PROFILE",
            "RDSP_P2P_TIMEOUT_S",
            "NCCL_SOCKET_IFNAME",
            "FI_PROVIDER",
            "HF_HOME",
            "PYTORCH_CUDA_ALLOC_CONF",
            "DS_BUILD_OPS",
            "CUDA_HOME",
            "LD_LIBRARY_PATH",
        )
    }


def test_credentials_stay_on_the_driver():
    # Ray stores a job's runtime_env with the job, where others can read it
    env = {
        "HF_TOKEN": "hf_x",
        "HF_HUB_TOKEN": "hf_y",
        "HUGGING_FACE_HUB_TOKEN": "hf_z",
        "RDSP_API_KEY": "k",
        "TORCH_SECRET": "s",
        "HF_HOME": "/hf",
        "HF_HUB_OFFLINE": "1",
    }
    assert train_vl.worker_env(env) == {"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}
