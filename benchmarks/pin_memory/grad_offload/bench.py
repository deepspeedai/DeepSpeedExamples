# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Async gradient CPU offload pin on/off (#8207). Reuses model-tensor ZeRO-3 offload."""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.path.join(os.path.dirname(_HERE), "model_tensor_offload", "bench.py")
os.execv(sys.executable, [sys.executable, _MODEL, "--zero-stage", "3", *sys.argv[1:]])
