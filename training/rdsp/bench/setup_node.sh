#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# Fresh Linux GPU node -> ready to run recipes/ and bench/common/runs.sh.
# Python 3.12 venv in ~/venv, torch 2.13, DeepSpeed at the pinned commit every
# result used, rdsp (editable), Qwen3.5 kernels, and the NeMo container for the
# Megatron side. Needs NVIDIA drivers, CUDA, Docker with the NVIDIA runtime.
set -euxo pipefail
RDSP=$(cd "$(dirname "$0")/.." && pwd)
unset LD_LIBRARY_PATH                   # some images' CUDA libs break torch's cuDNN
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
uv venv -p 3.12 ~/venv
. ~/venv/bin/activate
# Versions as installed for the October runs (2B, 4B, 27B one node, 35B). The 9B MegatronMIMO
# runs' node had ray 2.60.0, flash-linear-attention 0.6.0 and liger-kernel 0.8.5 instead.
uv pip install "torch==2.13.0" "torchvision==0.28.0" "transformers==5.17.0" "ray==2.59.0" \
    cloudpickle "accelerate==1.15.0" safetensors datasets pillow ninja packaging numpy pytest
CUDA_HOME=/usr/local/cuda-$(python -c "import torch; print(torch.version.cuda)")
[ -d "$CUDA_HOME" ] || CUDA_HOME=/usr/local/cuda   # no toolkit matching torch's CUDA
echo "export CUDA_HOME=$CUDA_HOME" > ~/cuda_env.sh
export CUDA_HOME DS_BUILD_OPS=0
uv pip install --no-build-isolation \
    "git+https://github.com/deepspeedai/DeepSpeed.git@53a2ac44fb664bea838df3981ba4366b91643070"
uv pip install -e "$RDSP"
# Qwen3.5 linear attention, fused kernels
uv pip install "flash-linear-attention==0.5.2" "liger-kernel==0.8.4"
uv pip install --no-build-isolation "causal-conv1d==1.7.0"
mkdir -p ~/hf ~/mimo ~/runs
sudo docker pull nvcr.io/nvidia/nemo:26.08
python -c "import torch, deepspeed, transformers, ray; print('torch', torch.__version__, \
'gpus', torch.cuda.device_count(), 'deepspeed', deepspeed.__version__, \
'transformers', transformers.__version__, 'ray', ray.__version__)"
