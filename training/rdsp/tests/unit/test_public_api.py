# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""The public API: what a user imports, and the contract each name states."""

from types import SimpleNamespace

import pytest
from test_partition import ToyLM

import ray_deepspeed_pipeline as rdsp
from ray_deepspeed_pipeline.compiler import lower
from ray_deepspeed_pipeline.errors import ValidationError
from ray_deepspeed_pipeline.partition import vision_token_ratio

DS = {"gradient_accumulation_steps": 2, "train_batch_size": 8}


def test_layout_types_are_public():
    assert {"StageOverride", "ConnectionOverride"} <= set(rdsp.__all__)


def _stated(obj) -> bool:
    """A written docstring, not dataclass's generated `Name(field: type, ...)`."""
    doc = (obj.__doc__ or "").strip()
    return bool(doc) and not doc.startswith(f"{getattr(obj, '__name__', '')}(")


def test_every_public_name_states_its_contract():
    for name in rdsp.__all__:
        assert _stated(getattr(rdsp, name)), name
    for method in (
        "train_batch",
        "eval_batch",
        "save_checkpoint",
        "load_checkpoint",
        "stage_blocks",
    ):
        assert _stated(getattr(rdsp.RayPipelineEngine, method)), method


def test_tp_must_divide_key_value_heads():
    """TP splits attention heads across GPUs; 2 key/value heads cannot be
    split 4 ways. Caught before any actor starts."""
    model = ToyLM()
    model.config = SimpleNamespace(num_key_value_heads=2)
    with pytest.raises(ValidationError, match="key/value heads"):
        lower(
            model,
            rdsp.PipelineConfig(
                stages=2,
                partition=rdsp.UniformTransformerBlocks(),
                stage_overrides=(rdsp.StageOverride(stage=0, num_gpus=4, tp=4),),
            ),
            DS,
        )


def test_vision_token_ratio_is_neutral_without_an_encoder():
    assert vision_token_ratio(SimpleNamespace(), []) == 1.0
