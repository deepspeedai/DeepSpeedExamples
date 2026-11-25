# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""ExecutionPlan canonical serialization and hashing are stable."""

import dataclasses
import json

import pytest

from ray_deepspeed_pipeline.plan import (
    SCHEMA_VERSION,
    ExecutionPlan,
    ScheduleSpec,
    StageConnection,
    StageSpec,
)


def make_plan(microbatches=4):
    stages = tuple(
        StageSpec(
            index=i,
            parameter_names=(f"blocks.{i}.fc.weight",),
            block_start=i,
            block_stop=i + 1,
            num_gpus=1,
            ds_config_json="{}",
            process_group_tag=f"stage{i}",
        )
        for i in range(2)
    )
    return ExecutionPlan(
        schema_version=SCHEMA_VERSION,
        global_microbatches=microbatches,
        stages=stages,
        connections=(StageConnection(source=0, dest=1, conversion="identity"),),
        schedule=ScheduleSpec(kind="1f1b", global_microbatches=microbatches),
    )


def test_canonical_json_is_deterministic():
    assert make_plan().canonical_json() == make_plan().canonical_json()
    parsed = json.loads(make_plan().canonical_json())
    assert parsed["schema_version"] == SCHEMA_VERSION
    assert parsed["stages"][0]["parameter_names"] == ["blocks.0.fc.weight"]


def test_hash_stable_and_content_sensitive():
    assert make_plan().plan_hash() == make_plan().plan_hash()
    assert make_plan(4).plan_hash() != make_plan(8).plan_hash()


def test_plan_is_frozen_plain_data():
    plan = make_plan()
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.global_microbatches = 99
