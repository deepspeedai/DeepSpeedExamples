# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Run all steps with default settings:
$ python3 e2e_rlhf.py

Change the model used for each step:
$ python3 e2e_rlhf.py --actor-model 350m --reward-model 1.3b

Change the ZeRO stage used for actor/reward models:
$ python3 e2e_rlhf.py --actor-zero-stage 1 --reward-zero-stage 3

Run a subset of the steps:
$ python3 e2e_rlhf.py --step 1 2

Note: Step 3 relies on models trained in Steps 1 & 2. If you have already
trained these models, you can run just Step 3 and select which models from
Steps 1 & 2 to use. For example, let's train models for Steps 1 & 2 using
125m and 350m models:
$ python3 e2e_rlhf.py --step 1 2 --actor-model 125m --reward-model 125m
$ python3 e2e_rlhf.py --step 1 2 --actor-model 350m --reward-model 350m

Now we can run Step 3 with any combination of these models:
$ python3 e2e_rlhf.py --step 3 --actor-model 125m --reward-model 350m
$ python3 e2e_rlhf.py --step 3 --actor-model 350m --reward-model 125m
"""

import argparse
import warnings
import subprocess
import os
import datetime
import time
import sys

step_dirs = {
    1: "training/step1_supervised_finetuning",
    2: "training/step2_reward_model_finetuning",
    3: "training/step3_rlhf_finetuning",
}
model_type = {1: "actor", 2: "reward", 3: "step3"}
dse_url = "https://github.com/deepspeedai/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        type=int,
        nargs="+",
        choices=(1, 2, 3),
        default=(1, 2, 3),
        help="Which steps of the ChatGPT example to run",
    )
    parser.add_argument(
        "--actor-model",
        type=lambda x: x.replace("facebook/opt-", ""),
        default="1.3b",
        choices=("1.3b", "6.7b", "13b", "66b"),
        help="Which facebook/opt-* model to use for Actor (step 1)",
    )
    parser.add_argument(
        "--reward-model",
        type=lambda x: x.replace("facebook/opt-", ""),
        default="350m",
        choices=("350m", ),
        help="Which facebook/opt-* model to use for Reward (step 2)",
    )
    parser.add_argument(
        "--actor-zero-stage",
        type=str,
        default="",
        choices=("", "0", "1", "2", "3"),
        help="ZeRO stage for step 1 (Actor) training",
    )
    parser.add_argument(
        "--reward-zero-stage",
        type=str,
        default="",
        choices=("", "0", "1", "2", "3"),
        help="ZeRO stage for step 2 (Critic) training",
    )
    parser.add_argument(
        "--output-dir",
        type=lambda x: os.path.abspath(x),
        default="./output",
        help="Directory for output of each step",
    )
    parser.add_argument(
        "--deployment-type",
        type=str,
        default="single_gpu",
        choices=("single_gpu", "single_node", "multi_node"),
        help="Number of GPUs to run the actor/reward models on",
    )
    args = parser.parse_args()

    if args.actor_zero_stage != "" or args.reward_zero_stage != "":
        warnings.warn(
            "Non-default zero stages may result in OOM errors or worse performance."
        )

    return args


def get_model_size(args, step_num):
    if step_num == 3:
        return get_model_size(args, 1)
    return getattr(args, f"{model_type[step_num]}_model")


def get_zero_stage(args, step_num):
    return getattr(args, f"{model_type[step_num]}_zero_stage")


def get_output_dir(args, step_num):
    model_size = get_model_size(args, step_num)
    output_dir = os.path.join(args.output_dir,
                              f"{model_type[step_num]}-models",
                              f"{model_size}")
    return output_dir


def get_script(args, step_num):
    model_size = get_model_size(args, step_num)
    script = os.path.join(
        os.getcwd(),
        step_dirs[step_num],
        "training_scripts/opt/",
        args.deployment_type,
        f"run_{model_size}.sh",
    )
    assert os.path.isfile(
        script
    ), f"{script} does not exist.\n\n Use examples in {os.path.dirname(script)} as a template."

    return script


def verify_model(args, step_num):
    output_dir = get_output_dir(args, step_num)
    model_size = get_model_size(args, step_num)
    model_file = os.path.join(output_dir, "pytorch_model.bin")
    if not os.path.isfile(model_file):
        error_str = f"Step {step_num} model has not been trained. Train it with:\n"
        error_str += f"{sys.executable.split('/')[-1]} {sys.argv[0]} --step {step_num}"
        error_str += f" --{model_type[step_num]}-model {model_size}"
        raise RuntimeError(error_str)


def get_cmd(args, step_num):
    output_dir = get_output_dir(args, step_num)
    script = get_script(args, step_num)

    if step_num in (1, 2):
        zero_stage = get_zero_stage(args, step_num)
        cmd = f"bash {script} {output_dir} {zero_stage}"
    if step_num == 3:
        verify_model(args, 1)  # Verify step 1 model exists
        verify_model(args, 2)  # Verify step 2 model exists
        s1_dir, s1_zs = get_output_dir(args, 1), get_zero_stage(args, 1)
        s2_dir, s2_zs = get_output_dir(args, 2), get_zero_stage(args, 2)
        cmd = f"bash {script} {s1_dir} {s2_dir} '{s1_zs}' '{s2_zs}' {output_dir}"

    return cmd


def launch_cmd(args, step_num, cmd):
    working_dir = step_dirs[step_num]
    print(f"Running:\n{cmd}")
    p = subprocess.Popen(cmd, cwd=working_dir, shell=True)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError('\n\n'.join((
            f"Step {step_num} exited with non-zero status {p.returncode}",
            f"Launch command: {cmd}",
            f"Log output: {os.path.join(get_output_dir(args, step_num), 'training.log')}",
            f"Please see our tutorial at {dse_url}{step_dirs[step_num]}",
            "Please check that you have installed our requirements: `pip install -r requirements.txt`",
            f"If you are seeing an OOM error, try modifying {get_script(args, step_num)}:",
            "  - Reduce `--per_device_*_batch_size`",
            "  - Increase `--zero_stage {0,1,2,3}` on multi-gpu setups",
            "  - Enable `--gradient_checkpointing` or `--only_optimize_lora`"
        )))


def main(args):
    start_time = time.time()
    for step_num in args.step:
        print(f"---=== Running Step {step_num} ===---")
        step_start_time = time.time()

        cmd = get_cmd(args, step_num)
        launch_cmd(args, step_num, cmd)

        step_time = int(time.time() - step_start_time)
        time_str = str(datetime.timedelta(seconds=step_time))
        print(f"---=== Finished Step {step_num} in {time_str} ===---")

    total_time = int(time.time() - start_time)
    time_str = str(datetime.timedelta(seconds=total_time))

    if len(args.step) > 1:
        print(f"---=== Finished Steps {args.step} in {time_str} ===---")


if __name__ == "__main__":
    args = parse_args()
    main(args)
