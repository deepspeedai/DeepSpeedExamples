# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import os
import subprocess


def file_exists(directory_path, file_name):
    return os.path.isfile(os.path.join(directory_path, file_name))


@pytest.fixture(params=["2", "3"], ids=["zero2", "zero3"])
def zero_stage(request):
    return str(request.param)


@pytest.fixture(params=["true", "false"], ids=["he", ""])
def hybrid_engine(request):
    return str(request.param)


@pytest.fixture(params=["true", "false"], ids=["offload", ""])
def offload(request):
    return str(request.param)


@pytest.fixture(params=["true", "false"], ids=["lora", ""])
def lora(request):
    return str(request.param)


def test_ds_chat(zero_stage, hybrid_engine, offload, lora):
    # Assert that critic model directory exists
    critic_ckpt_dir = os.getenv("CRITIC_CKPT_DIR")
    assert critic_ckpt_dir, "Please set CRITIC_CKPT_DIR in your environment"

    # Setup params
    actor_model = "facebook/opt-125m"
    critic_model = critic_ckpt_dir
    mixed_precision_lora = "false"
    enable_test_mode = "true"
    test_stop_step = "5"
    output_path = "z" + zero_stage + "_he_" + hybrid_engine + "_offload_" + offload + "_lora_" + lora
    params = [
        actor_model,
        critic_model,
        zero_stage,
        zero_stage,
        hybrid_engine,
        offload,
        lora,
        mixed_precision_lora,
        output_path,
        enable_test_mode,
        test_stop_step,
    ]

    # Skip certain combinations
    if zero_stage == "2" and hybrid_engine == "true" and offload == "true" and lora == "false":
        pytest.skip(
            "The combination of [actor_zero_stage==2, critic_zero_stage==2, enable_hybrid_engine=True, offload=True, lora=False] is currently unsupported due to training instability!"
        )

    # cd into execution dir
    wd = os.getcwd()
    os.chdir("../training/step3_rlhf_finetuning")
    sweep_script = "training_scripts/opt/single_node/sweep/run_single.sh"

    # Run bash script
    cmd = ["bash", sweep_script] + params
    result = subprocess.run(cmd)

    # Assertions
    try:
        result.check_returncode()
    except subprocess.CalledProcessError as e:
        with open(os.path.join(output_path, f"{output_path}.log"), "r") as f:
            print(f.read())
        raise e

    assert file_exists(f"{output_path}/actor/", "pytorch_model.bin"
                       ), "Actor model was not saved during step 3 training."
    assert file_exists(f"{output_path}/critic/", "pytorch_model.bin"
                       ), "Critic model was not saved during step 3 training."

    os.chdir(wd)
