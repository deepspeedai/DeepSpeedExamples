# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Stable public exception classes. No Ray or DeepSpeed subclassing."""


class RdspError(Exception):
    """Base class for all public ray_deepspeed_pipeline errors."""


class ValidationError(RdspError):
    """A public argument failed validation before any actor work."""


class UnsupportedInV1(ValidationError):
    """A deepspeed.initialize() argument form this package rejects (e.g. a
    preconstructed optimizer)."""


class UnsupportedEngineMethod(RdspError):
    """A local-engine method that pipeline mode deliberately does not provide."""


class StepFailed(RdspError):
    """A train/eval step failed before any optimizer update was applied.

    global_steps did not advance; the call may be retried."""


class PipelinePoisoned(RdspError):
    """Optimizer updates may have been partially applied across stages.

    Train/eval calls are rejected until load_checkpoint() restores the pipeline."""


class CheckpointError(RdspError):
    """A global checkpoint could not be committed or restored.

    A failed save publishes no manifest; a rejected load leaves worker state untouched."""
