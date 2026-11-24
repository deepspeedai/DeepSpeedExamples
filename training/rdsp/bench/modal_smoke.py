# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Check that the shared NGC benchmark image works for both frameworks.

    modal run bench/modal_smoke.py                  # from the repo root
    modal run bench/modal_smoke.py --only megatron  # Megatron PP=2 run only

Checks:
  1. the image builds and CUDA works on Modal's driver
  2. Megatron-Core, TransformerEngine, DeepSpeed (pinned), Ray and rdsp import
  3. pretrain_gpt.py knows the Qwen3 flags (--kv-channels, --qk-layernorm)
  4. rdsp's Qwen3-0.6B two-stage acceptance test passes in this image
  5. a 20-iteration Megatron PP=2 mock-data run completes
"""

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
NGC = "nvcr.io/nvidia/pytorch:25.10-py3"
MCORE_TAG = "core_v0.19.2"
DS_REV = "53a2ac44fb664bea838df3981ba4366b91643070"

image = (
    modal.Image.from_registry(NGC)
    .env(
        {
            "PIP_CONSTRAINT": "",
            "DS_BUILD_OPS": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": "/root/.cache/huggingface",
            # lets containers import bench modules (modal_bench imports this file)
            "PYTHONPATH": "/root/rdsp/bench",
        }
    )
    .run_commands(
        f"git clone --depth 1 --branch {MCORE_TAG} "
        "https://github.com/NVIDIA/Megatron-LM.git /opt/Megatron-LM",
        "pip install --no-build-isolation -e /opt/Megatron-LM",
        # 25.10 ships an nvidia-resiliency-ext without __version__, which
        # Megatron probes at import; it is optional (async checkpointing only)
        "pip uninstall -y nvidia-resiliency-ext",
        "pip install 'transformers==5.16.1' accelerate safetensors einops "
        "'ray[default]==2.58.0' cloudpickle nvidia-ml-py pytest cupy-cuda13x",
        # --no-deps: liger's triton pin would replace NGC's triton, which
        # breaks torch 2.9's inductor (Megatron-defaults torch.compile path)
        "pip install --no-deps liger-kernel",
        f"pip install --no-build-isolation git+https://github.com/deepspeedai/DeepSpeed.git@{DS_REV}",
    )
    .add_local_dir(
        REPO,
        "/root/rdsp",
        ignore=[".venv", "**/__pycache__", ".pytest_cache", "uv.lock", ".cache", "bench/results"],
    )
)
app = modal.App("rdsp-bench-smoke", image=image)
hf_cache = modal.Volume.from_name("rdsp-hf-cache", create_if_missing=True)


def _sh(cmd: str) -> tuple[int, str]:
    import subprocess

    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)[-6000:]


@app.function(
    gpu="H100:2", timeout=3600, cpu=16, memory=65536, volumes={"/root/.cache/huggingface": hf_cache}
)
def smoke():
    results = {}
    results["nvidia-smi"] = _sh("nvidia-smi --query-gpu=name,driver_version --format=csv")
    results["imports"] = _sh(
        'python -c "import torch, transformer_engine, megatron.core, deepspeed, ray; '
        "print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), "
        "transformer_engine.__version__, megatron.core.__version__, deepspeed.__version__, "
        'ray.__version__)"'
    )
    results["flags"] = _sh(
        "cd /opt/Megatron-LM && python pretrain_gpt.py --help 2>&1 | "
        "grep -E -- '--kv-channels|--qk-layernorm|"
        "--pipeline-model-parallel-layout' | head"
    )
    results["rdsp_install"] = _sh("pip install -e /root/rdsp")
    results["rdsp_acceptance"] = _sh(
        "cd /root/rdsp && python -m pytest -q -x "
        "tests/integration/test_two_stage_baseline.py 2>&1 | tail -5"
    )
    results["megatron_pp2"] = _sh(
        "cd /root/rdsp/bench && PP=2 M=4 SEQ=512 ITERS=20 VARIANT=plain "
        "MEGATRON=/opt/Megatron-LM bash megatron_pretrain.sh 2>&1 | "
        "grep -E 'elapsed time per iteration|lm loss|Error|error' | tail -8"
    )
    hf_cache.commit()
    return results


@app.function(
    gpu="H100:2", timeout=1800, cpu=16, memory=65536, volumes={"/root/.cache/huggingface": hf_cache}
)
def megatron_only(extra: str = ""):
    import subprocess

    r = subprocess.run(
        f"cd /root/rdsp/bench && PP=2 M=4 SEQ=512 ITERS=20 VARIANT=plain "
        f"MEGATRON=/opt/Megatron-LM {extra} bash megatron_pretrain.sh",
        shell=True,
        capture_output=True,
        text=True,
    )
    out = r.stdout + r.stderr
    keep = [
        line
        for line in out.splitlines()
        if "[rank0]" in line or "elapsed time" in line or "lm loss" in line
    ]
    return r.returncode, "\n".join(keep[-60:]) or out[-8000:]


@app.local_entrypoint()
def main(only: str = ""):
    if only == "megatron":
        code, out = megatron_only.remote()
        print(out, f"\nexit {code}")
        return
    for k, (code, out) in smoke.remote().items():
        print(f"===== {k} (exit {code})\n{out}")
