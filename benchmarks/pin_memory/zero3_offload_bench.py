# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Backward-compatible entry: model-tensor ZeRO-3 offload (see model_tensor_offload/)."""

from __future__ import annotations

import os
import sys

_NEW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_tensor_offload", "bench.py")
os.execv(sys.executable, [sys.executable, _NEW, "--zero-stage", "3", *sys.argv[1:]])
