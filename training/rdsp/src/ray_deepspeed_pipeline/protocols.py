# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Internal contracts: the engine/coordinator seam and the worker command protocol."""

import dataclasses
import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

# "step"/"eval_step" carry a stage's whole op list for one step (stage-local dispatch)
COMMAND_KINDS = (
    "forward",
    "backward",
    "ready",
    "apply",
    "eval",
    "save",
    "load",
    "step",
    "eval_step",
)


@dataclass(frozen=True)
class Command:
    """One unit of stage work, generated deterministically by schedule.py.

    Commands that run collectives must reach every rank of a stage in the same order.
    """

    command_id: str
    generation: int
    global_step: int
    stage: int
    kind: str  # one of COMMAND_KINDS
    microbatch: int | None
    predecessors: tuple[str, ...]

    def to_canonical_dict(self) -> dict:
        return dataclasses.asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_canonical_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class CommandResult:
    command_id: str
    status: str  # "ok" | "failed"
    detail: str = ""


class Coordinator(Protocol):
    """Runs one global step at a time and owns global_steps."""

    @property
    def global_steps(self) -> int: ...

    def train_batch(self, data_iter: Iterator) -> Any: ...

    def eval_batch(self, data_iter: Iterator) -> Any: ...

    def save_checkpoint(self, save_dir: str, tag: str | None, **kwargs) -> Any: ...

    def load_checkpoint(self, load_dir: str, tag: str | None, **kwargs) -> Any: ...


def resolve(result: Any) -> Any:
    """Return a plain value, calling .result() on a future-like result."""
    if hasattr(result, "result") and callable(result.result):
        return result.result()
    return result
