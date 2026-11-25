# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Internal command protocol schema and facade result resolution."""

import dataclasses
import json

import pytest

from ray_deepspeed_pipeline.protocols import (
    COMMAND_KINDS,
    Command,
    CommandResult,
    resolve,
)


def test_command_canonical_roundtrip():
    cmd = Command(
        command_id="g0.s1.f2",
        generation=0,
        global_step=3,
        stage=1,
        kind="forward",
        microbatch=2,
        predecessors=("g0.s0.f2",),
    )
    parsed = json.loads(cmd.canonical_json())
    assert parsed == {
        "command_id": "g0.s1.f2",
        "generation": 0,
        "global_step": 3,
        "stage": 1,
        "kind": "forward",
        "microbatch": 2,
        "predecessors": ["g0.s0.f2"],
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        cmd.kind = "backward"


def test_command_kinds_cover_the_runtime_verbs():
    assert set(COMMAND_KINDS) == {
        "forward",
        "backward",
        "ready",
        "apply",
        "eval",
        "save",
        "load",
        # stage-local dispatch: a stage's whole step
        "step",
        "eval_step",
    }


def test_result_schema():
    r = CommandResult(command_id="x", status="ok")
    assert r.detail == ""


def test_resolve_unwraps_future_like_and_passes_values():
    class Fut:
        def result(self):
            return 41

    assert resolve(Fut()) == 41
    assert resolve(0.5) == 0.5
    assert resolve(None) is None
