#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# On the H100 box: TorchTitan at the commit read and checked (3c36a77) in ~/torchtitan, its own
# venv ~/tt_venv (Python 3.12, the PyTorch nightly for CUDA 13.0 of the date of that commit; the
# setup fails if that nightly is gone), TorchTitan's requirements + its vision extras. Leaves the
# rdsp venv alone.
set -euxo pipefail
export PATH=$HOME/.local/bin:$PATH
unset LD_LIBRARY_PATH
[ -d ~/torchtitan ] || git clone https://github.com/pytorch/torchtitan ~/torchtitan
git -C ~/torchtitan checkout 3c36a7767cf2222aa4e0d494463f486363c8df99
uv venv -p 3.12 ~/tt_venv
. ~/tt_venv/bin/activate
NIGHTLY=https://download.pytorch.org/whl/nightly/cu130
uv pip install --pre "torch==2.16.0.dev20261012" "torchvision==0.30.0.dev20261012" --index-url $NIGHTLY
uv pip install -r ~/torchtitan/requirements.txt -r ~/torchtitan/.ci/docker/requirements-vlm.txt
uv pip install --no-deps -e ~/torchtitan
python -c "import torch, torchtitan; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"
echo "TT SETUP DONE"
