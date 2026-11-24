# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""ExecutionPlan: the normalized plan every PipelineConfig lowers to.

The runtime reads only this, never PipelineConfig. Plain, JSON-canonical data
so the plan has a stable hash; checkpoints are bound to that hash.
"""

import dataclasses
import hashlib
import json
from dataclasses import dataclass

SCHEMA_VERSION = "4"


@dataclass(frozen=True)
class StageSpec:
    index: int
    # fully qualified names in model order; the actor builds its module from them
    parameter_names: tuple[str, ...]
    # contiguous block range [block_start, block_stop) within the model's
    # transformer block list; first/last stage also own the pre/post modules
    block_start: int
    block_stop: int
    num_gpus: int
    # canonical JSON of this stage's ordinary DeepSpeed config
    ds_config_json: str
    process_group_tag: str
    # stage-local rank grid (num_gpus = dp * sp * tp); ep folds onto it
    dp: int = 1
    tp: int = 1
    sp: int = 1
    ep: int = 1
    fold: bool = False
    # rows of one global microbatch each dp shard holds
    rows_per_rank: int = 1
    # rerun blocks in backward instead of keeping their activations
    recompute: bool = False
    # torch.compile each block, or only the vision encoder's
    compile: bool = False
    compile_vision: bool = False
    # recompute only the first N blocks (and the encoder's); None: per `recompute`
    recompute_blocks: int | None = None


@dataclass(frozen=True)
class StageConnection:
    source: int
    dest: int
    # identity | replicate-to-shard | shard-to-replicate | shard-to-shard
    conversion: str


@dataclass(frozen=True)
class ScheduleSpec:
    kind: str  # "1f1b"
    global_microbatches: int


@dataclass(frozen=True)
class VisionSpec:
    """Colocated vision encoder: its module path in the model, its
    parameters (on no stage), and whether its blocks recompute or compile."""

    module: str
    parameter_names: tuple[str, ...]
    recompute: bool = False
    compile: bool = False
    encode_per_microbatch: bool = False


# fields that change how a step runs, not a stage's parameters or saved state
_RUNTIME_ONLY = frozenset(
    {
        "prefetch",
        "recompute",
        "recompute_blocks",
        "compile",
        "compile_vision",
        "encode_per_microbatch",
    }
)


def _without(value, keys: frozenset):
    if isinstance(value, dict):
        return {k: _without(v, keys) for k, v in value.items() if k not in keys}
    if isinstance(value, (list, tuple)):
        return [_without(v, keys) for v in value]
    return value


@dataclass(frozen=True)
class ExecutionPlan:
    schema_version: str
    global_microbatches: int
    stages: tuple[StageSpec, ...]
    connections: tuple[StageConnection, ...]
    schedule: ScheduleSpec
    # rows in one global microbatch (the data loader's entry size)
    microbatch_rows: int = 1
    # vision encoder on every rank instead of the first stage (vision.py)
    colocated_vision: VisionSpec | None = None
    # read and ship the next training step's data while this one runs
    prefetch: bool = False

    def to_canonical_dict(self) -> dict:
        return dataclasses.asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_canonical_dict(), sort_keys=True, separators=(",", ":"))

    def plan_hash(self) -> str:
        """Binds checkpoints to the plan. Settings that change how steps run
        but not what a stage saves (_RUNTIME_ONLY) are left out, so toggling
        them keeps a checkpoint loadable."""
        kept = _without(self.to_canonical_dict(), _RUNTIME_ONLY)
        return hashlib.sha256(
            json.dumps(kept, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
