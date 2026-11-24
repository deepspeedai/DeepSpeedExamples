# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""The public initialize() signature is frozen."""

import inspect

import ray_deepspeed_pipeline as rdsp

EXPECTED_PARAMS = [
    ("model", inspect.Parameter.empty),
    ("optimizer", None),
    ("model_parameters", None),
    ("training_data", None),
    ("lr_scheduler", None),
    ("collate_fn", None),
    ("config", None),
    ("pipeline_config", inspect.Parameter.empty),
    ("loss_fn", inspect.Parameter.empty),
    ("weights", None),
]


def test_signature_names_defaults_and_keyword_only():
    sig = inspect.signature(rdsp.initialize)
    params = list(sig.parameters.values())
    assert [(p.name, p.default) for p in params] == EXPECTED_PARAMS
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params), (
        "every initialize() parameter must be keyword-only"
    )


def test_public_exports():
    assert set(rdsp.__all__) >= {"initialize", "PipelineConfig", "RayPipelineEngine"}
