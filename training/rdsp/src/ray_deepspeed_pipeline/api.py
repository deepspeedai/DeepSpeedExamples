# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Public initialize(): argument validation and engine assembly."""

import json

import cloudpickle
import torch

from ray_deepspeed_pipeline.config import PipelineConfig
from ray_deepspeed_pipeline.data import build_training_dataloader
from ray_deepspeed_pipeline.engine import RayPipelineEngine
from ray_deepspeed_pipeline.errors import UnsupportedInV1, ValidationError

# Pipeline/runtime settings belong in PipelineConfig, never in the DeepSpeed config.
_RESERVED_CONFIG_KEYS = frozenset(
    {
        "pipeline",
        "pipeline_config",
        "ray",
        "actors",
        "placement",
        "stages",
        "schedule",
        "transport",
        "manifest",
        "connections",
    }
)


def _default_coordinator_factory(*, model, pipeline_config, ds_config, loss_fn, weights):
    """Lower to an ExecutionPlan and start stage actor groups in the current
    Ray context (never initializing one)."""
    from ray_deepspeed_pipeline.compiler import lower
    from ray_deepspeed_pipeline.coordinator import PipelineCoordinator
    from ray_deepspeed_pipeline.stage_group import create_stage_clients

    config = ds_config if isinstance(ds_config, dict) else {}
    plan = lower(model, pipeline_config, config)
    coordinator = PipelineCoordinator(
        plan, create_stage_clients(model, plan, loss_fn, weights=weights)
    )

    def rebuild():
        # prefer each stage's previous node: its checkpoint shards may be node-local
        previous_nodes = [getattr(w, "node_id", None) for w in coordinator._workers]
        return create_stage_clients(
            model, plan, loss_fn, weights=weights, prefer_nodes=previous_nodes
        )

    coordinator._rebuild = rebuild
    return coordinator


# Test seam: contract tests swap in a fake coordinator.
_coordinator_factory = _default_coordinator_factory


def _validate_config(config):
    if config is None:
        return None
    if not isinstance(config, (dict, str)):
        raise ValidationError(
            f"config must be an ordinary DeepSpeed config mapping or path, got {type(config)}"
        )
    if isinstance(config, str):
        # loaded here so planning sees the real settings (microbatches, batch size)
        try:
            with open(config) as f:
                config = json.load(f)
        except (OSError, ValueError) as e:
            raise ValidationError(f"cannot read DeepSpeed config {config!r}: {e}") from e
        if not isinstance(config, dict):
            raise ValidationError("a DeepSpeed config file must contain a JSON object")
    bad = _RESERVED_CONFIG_KEYS.intersection(config)
    if bad:
        raise ValidationError(
            f"config must remain an ordinary DeepSpeed configuration; "
            f"external-runtime field(s) {sorted(bad)} belong in pipeline_config"
        )
    return config


def _validate_loss_fn(loss_fn):
    if not callable(loss_fn):
        raise ValidationError("loss_fn must be a callable loss_fn(outputs, labels)")
    try:
        cloudpickle.dumps(loss_fn)
    except Exception as e:
        raise ValidationError(
            f"loss_fn must be serializable (it is shipped to the terminal stage actor): {e}"
        ) from e


def _check_model_parameters(model, model_parameters) -> None:
    """Accept None or exactly the model's own parameters (the usual
    deepspeed.initialize idiom). Each stage builds its optimizer over its own
    slice of the model, so subsets and parameter groups cannot be honoured."""
    if model_parameters is None:
        return
    params = list(model_parameters)
    if any(isinstance(p, dict) for p in params):
        raise UnsupportedInV1(
            "parameter groups are not supported: each stage builds its optimizer "
            "from the DeepSpeed config over its own parameters"
        )
    if not all(isinstance(p, torch.nn.Parameter) for p in params):
        raise ValidationError("model_parameters must contain torch.nn.Parameter objects")
    ids = [id(p) for p in params]
    if len(set(ids)) != len(ids):
        raise ValidationError("model_parameters lists a parameter twice")
    if set(ids) != {id(p) for p in model.parameters()}:
        raise UnsupportedInV1(
            "model_parameters must be None or all of model.parameters(): each "
            "stage optimizes every parameter it owns, so subsets cannot be honoured"
        )


def initialize(
    *,
    model,
    optimizer=None,
    model_parameters=None,
    training_data=None,
    lr_scheduler=None,
    collate_fn=None,
    config=None,
    pipeline_config,
    loss_fn,
    weights=None,
):
    """deepspeed.initialize() for a pipeline of stage actors.

    Returns (engine, None, training_dataloader_or_None, None): the optimizer and
    scheduler live inside the stage actors, so they are never returned.
    weights: an HF checkpoint directory; model parameters still on the meta
    device are loaded from it by each stage, only the stage's own tensors.

    Raises ValidationError (bad config, unsupported layout, TP that does not
    divide the key/value heads, a stage that cannot be placed on the Ray
    cluster) or UnsupportedInV1 (a DeepSpeed feature rdsp does not run)
    before training starts."""
    if optimizer is not None:
        raise UnsupportedInV1(
            "optimizer must be None in v1: a driver-side optimizer object "
            "cannot be rebound to stage-local parameters inside actors. "
            "Define the optimizer in the DeepSpeed config; each stage "
            "constructs it locally via ordinary deepspeed.initialize()"
        )
    if lr_scheduler is not None:
        raise UnsupportedInV1(
            "lr_scheduler must be None in v1: define the scheduler in the "
            "DeepSpeed config; each stage constructs it locally"
        )
    if not isinstance(model, torch.nn.Module):
        raise ValidationError(f"model must be a torch.nn.Module, got {type(model)}")
    if not isinstance(pipeline_config, PipelineConfig):
        raise ValidationError(
            f"pipeline_config must be a PipelineConfig, got {type(pipeline_config)}"
        )
    if pipeline_config.prefetch and training_data is not None:
        raise ValidationError(
            "prefetch needs the caller's iterator (train_batch(data_iter=...)): with "
            "training_data the engine counts batches for checkpoints, and the step "
            "read ahead would be counted before it ran"
        )

    ds_config = _validate_config(config)
    _validate_loss_fn(loss_fn)
    _check_model_parameters(model, model_parameters)

    training_dataloader = None
    if training_data is not None:
        from ray_deepspeed_pipeline.compiler import (
            global_microbatch_rows,
            resolve_microbatches,
        )

        conf = ds_config if isinstance(ds_config, dict) else None
        try:
            rows = global_microbatch_rows(conf, resolve_microbatches(pipeline_config, conf))
        except ValidationError:
            # lowering reports this error itself, before any actor exists
            rows = int((conf or {}).get("train_micro_batch_size_per_gpu", 1))
        training_dataloader = build_training_dataloader(training_data, collate_fn, rows)

    coordinator = _coordinator_factory(
        model=model,
        pipeline_config=pipeline_config,
        ds_config=ds_config,
        loss_fn=loss_fn,
        weights=weights,
    )
    engine = RayPipelineEngine(coordinator, training_dataloader)
    return engine, None, training_dataloader, None
