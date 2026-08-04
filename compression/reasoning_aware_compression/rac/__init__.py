# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Reasoning-Aware Compression (RAC): one-shot pruning of reasoning LLMs.

Implements `Reasoning Models Can be Accurately Pruned Via Chain-of-Thought
Reconstruction <https://arxiv.org/abs/2509.12464>`_ (Lucas et al., ICLR 2026).
"""

from .calibration import CALIBRATION_SOURCES, build_calibration_samples
from .magnitude import magnitude_prune
from .modelutils import (
    dtype_kwargs,
    find_linear_layers,
    get_decoder_layers,
    layerwise_sparsity,
    model_sparsity,
    prunable_layers,
    select_blocks_by_thirds,
)
from .sequential import sequential_prune
from .sparsegpt import SparseGPT
from .wanda import Wanda

__all__ = [
    "CALIBRATION_SOURCES",
    "SparseGPT",
    "Wanda",
    "build_calibration_samples",
    "dtype_kwargs",
    "find_linear_layers",
    "get_decoder_layers",
    "layerwise_sparsity",
    "magnitude_prune",
    "model_sparsity",
    "prunable_layers",
    "select_blocks_by_thirds",
    "sequential_prune",
]
