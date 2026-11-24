# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Run rdsp test files on Modal GPUs, with DeepSpeed from the pinned revision.

From the repo root:

    modal run scripts/modal_tests.py --gpus L4:2 --tests tests/integration/test_checkpoint_gpu.py
    modal run scripts/modal_tests.py --gpus L4:4 \
        --tests "tests/integration/test_heterogeneous_pipeline.py -k dp-zero"

One function per GPU shape: Modal fixes the GPU request at decoration time.
"""

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]

ORACLE_REV = "53a2ac44fb664bea838df3981ba4366b91643070"

hf_cache = modal.Volume.from_name("rdsp-hf-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        "torch",
        "pytest",
        "ninja",
        "packaging",
        "numpy",
        "transformers",
        "ray",
        "cloudpickle",
        "accelerate",
        "cupy-cuda12x",
    )  # cupy: Ray Direct Transport over NCCL
    .run_commands(
        "pip install --no-build-isolation "
        f"git+https://github.com/deepspeedai/DeepSpeed.git@{ORACLE_REV}"
    )
    .env({"DS_BUILD_OPS": "0", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(
        REPO,
        "/root/rdsp",
        ignore=[".venv", "**/__pycache__", ".cache", ".pytest_cache", "uv.lock", "bench/results"],
    )
)

app = modal.App("rdsp-tests", image=image)
VOLUMES = {"/root/.cache/huggingface": hf_cache}


def _run(tests: str) -> tuple[int, str]:
    """Run pytest, streaming output live (visible in `modal app logs`); return
    the exit code and the output tail."""
    import shlex
    import subprocess
    import sys

    subprocess.run(["pip", "install", "-e", "/root/rdsp"], check=True, capture_output=True)
    proc = subprocess.Popen(
        [
            "python",
            "-u",
            "-m",
            "pytest",
            "-v",
            "-s",
            "--tb=short",
            "-rA",
            "-o",
            "faulthandler_timeout=600",
            *shlex.split(tests),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd="/root/rdsp",
    )
    lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line)
    proc.wait()
    hf_cache.commit()
    return proc.returncode, "".join(lines)[-40000:]


@app.function(gpu="L4:2", timeout=3600, volumes=VOLUMES)
def run_l4x2(tests: str):
    return _run(tests)


@app.function(gpu="L4:4", timeout=3600, volumes=VOLUMES)
def run_l4x4(tests: str):
    return _run(tests)


@app.function(gpu="L4:8", timeout=3600, volumes=VOLUMES)
def run_l4x8(tests: str):
    return _run(tests)


@app.function(gpu="H100:8", timeout=3600, volumes=VOLUMES)
def run_h100x8(tests: str):
    return _run(tests)


RUNNERS = {"L4:2": run_l4x2, "L4:4": run_l4x4, "L4:8": run_l4x8, "H100:8": run_h100x8}


@app.local_entrypoint()
def main(gpus: str = "L4:2", tests: str = "tests/integration/test_checkpoint_gpu.py"):
    code, output = RUNNERS[gpus].remote(tests)
    print(output)
    print(f"\nexit code: {code}")
