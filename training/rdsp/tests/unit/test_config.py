# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""PipelineConfig schema is frozen and validating."""

import dataclasses

import pytest

import ray_deepspeed_pipeline as rdsp
from ray_deepspeed_pipeline.config import (
    ConnectionOverride,
    ExplicitCuts,
    PipelineConfig,
    StageOverride,
    UniformTransformerBlocks,
)
from ray_deepspeed_pipeline.errors import ValidationError


def test_schema_fields_frozen():
    fields = [f.name for f in dataclasses.fields(PipelineConfig)]
    assert fields == [
        "stages",
        "partition",
        "schedule",
        "microbatches",
        "stage_overrides",
        "connection_overrides",
        "colocated_vision",
        "prefetch",
    ]
    cfg = PipelineConfig(stages=2, partition=UniformTransformerBlocks())
    assert cfg.schedule == "1f1b" and cfg.microbatches is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.stages = 3


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(stages=0, partition=UniformTransformerBlocks()),
        dict(stages=2, partition=UniformTransformerBlocks(), schedule="gpipe"),
        dict(stages=2, partition=UniformTransformerBlocks(), microbatches=0),
        dict(stages=2, partition="uniform"),
        dict(
            stages=2,
            partition=UniformTransformerBlocks(),
            stage_overrides=(StageOverride(stage=5),),
        ),
        dict(
            stages=2,
            partition=UniformTransformerBlocks(),
            stage_overrides=(StageOverride(stage=0, num_gpus=0),),
        ),
        dict(
            stages=2,
            partition=UniformTransformerBlocks(),
            connection_overrides=("not-an-override",),
        ),
    ],
)
def test_invalid_configs_rejected(kwargs):
    with pytest.raises(ValidationError):
        PipelineConfig(**kwargs)


def test_explicit_cuts_normalizes_to_tuple():
    assert ExplicitCuts(cuts=[2, 4]).cuts == (2, 4)


def test_connection_override_defaults():
    c = ConnectionOverride(source=0, dest=1)
    assert c.conversion is None  # None: derived from layouts
    assert [f.name for f in dataclasses.fields(c)] == ["source", "dest", "conversion"]


def test_deepspeed_config_path_is_loaded(tmp_path, monkeypatch, tiny_model):
    """A config given as a JSON path reaches planning as its contents."""
    import json

    import ray_deepspeed_pipeline as rdsp
    from ray_deepspeed_pipeline import api

    seen = {}
    monkeypatch.setattr(api, "_coordinator_factory", lambda **kw: seen.update(kw) or object())
    path = tmp_path / "ds.json"
    path.write_text(json.dumps({"gradient_accumulation_steps": 4}))
    rdsp.initialize(
        model=tiny_model,
        config=str(path),
        loss_fn=lambda o, y: o.sum(),
        pipeline_config=rdsp.PipelineConfig(stages=2, partition=rdsp.UniformTransformerBlocks()),
    )
    assert seen["ds_config"] == {"gradient_accumulation_steps": 4}


@pytest.mark.parametrize(
    "content,match",
    [
        ('{"pipeline": {}}', "external-runtime"),
        ("[1, 2]", "JSON object"),
        ("{not json", "cannot read"),
    ],
)
def test_bad_deepspeed_config_file_rejected(tmp_path, tiny_model, content, match):
    import ray_deepspeed_pipeline as rdsp

    path = tmp_path / "ds.json"
    path.write_text(content)
    with pytest.raises(ValidationError, match=match):
        rdsp.initialize(
            model=tiny_model,
            config=str(path),
            loss_fn=lambda o, y: o.sum(),
            pipeline_config=rdsp.PipelineConfig(
                stages=2, partition=rdsp.UniformTransformerBlocks()
            ),
        )


def test_prefetch_needs_a_caller_iterator():
    """With training_data the engine counts batches handed out for
    checkpoints; a step read ahead would be counted before it ran."""
    import torch.nn as nn

    with pytest.raises(ValidationError, match="prefetch"):
        rdsp.initialize(
            model=nn.Linear(2, 2),
            config={"train_batch_size": 2, "gradient_accumulation_steps": 2},
            loss_fn=lambda o, y: o.sum(),
            training_data=[(1, 2)] * 4,
            pipeline_config=PipelineConfig(
                stages=2, partition=UniformTransformerBlocks(), prefetch=True
            ),
        )
