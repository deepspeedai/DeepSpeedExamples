# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""initialize() returns an honest 4-tuple and validates inputs before any actor starts."""

import pytest
import torch
from conftest import simple_loss_fn
from torch.utils.data import DataLoader

import ray_deepspeed_pipeline as rdsp
from ray_deepspeed_pipeline.errors import UnsupportedInV1, ValidationError


def _init(model, pipeline_config, **overrides):
    kwargs = dict(model=model, pipeline_config=pipeline_config, loss_fn=simple_loss_fn)
    kwargs.update(overrides)
    return rdsp.initialize(**kwargs)


def test_four_tuple_with_training_data(fake_coordinator, tiny_model, tiny_dataset, pipeline_config):
    result = _init(tiny_model, pipeline_config, training_data=tiny_dataset)
    assert isinstance(result, tuple) and len(result) == 4
    engine, optimizer, dataloader, scheduler = result
    assert isinstance(engine, rdsp.RayPipelineEngine)
    assert optimizer is None, "no fake optimizer proxy may be returned"
    assert isinstance(dataloader, DataLoader), (
        "third element is the driver-side loader, not an actor proxy"
    )
    assert scheduler is None, "no fake scheduler proxy may be returned"


def test_third_element_none_without_training_data(fake_coordinator, tiny_model, pipeline_config):
    _, _, dataloader, _ = _init(tiny_model, pipeline_config, training_data=None)
    assert dataloader is None


def test_preconstructed_optimizer_rejected(fake_coordinator, tiny_model, pipeline_config):
    opt = torch.optim.AdamW(tiny_model.parameters())
    with pytest.raises(UnsupportedInV1):
        _init(tiny_model, pipeline_config, optimizer=opt)


def test_preconstructed_scheduler_rejected(fake_coordinator, tiny_model, pipeline_config):
    opt = torch.optim.AdamW(tiny_model.parameters())
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    with pytest.raises(UnsupportedInV1):
        _init(tiny_model, pipeline_config, lr_scheduler=sched)


def test_foreign_parameters_rejected(fake_coordinator, tiny_model, pipeline_config):
    other = torch.nn.Linear(3, 3)
    with pytest.raises(ValidationError):
        _init(tiny_model, pipeline_config, model_parameters=other.parameters())


def test_duplicate_parameters_rejected(fake_coordinator, tiny_model, pipeline_config):
    p = next(tiny_model.parameters())
    with pytest.raises(ValidationError):
        _init(tiny_model, pipeline_config, model_parameters=[p, p])


def test_opaque_parameters_rejected(fake_coordinator, tiny_model, pipeline_config):
    with pytest.raises(ValidationError):
        _init(tiny_model, pipeline_config, model_parameters=[object()])


def test_all_model_parameters_accepted(fake_coordinator, tiny_model, pipeline_config):
    _init(tiny_model, pipeline_config, model_parameters=tiny_model.parameters())


def test_parameter_subsets_and_groups_rejected(fake_coordinator, tiny_model, pipeline_config):
    """Each stage optimizes every parameter it owns; anything narrower would be
    silently ignored, so it is refused."""
    with pytest.raises(UnsupportedInV1, match="subsets"):
        _init(tiny_model, pipeline_config, model_parameters=tiny_model[0].parameters())
    groups = [
        {"params": list(tiny_model[0].parameters()), "lr": 0.1},
        {"params": list(tiny_model[2].parameters())},
    ]
    with pytest.raises(UnsupportedInV1, match="groups"):
        _init(tiny_model, pipeline_config, model_parameters=groups)


def test_config_rejects_external_runtime_fields(fake_coordinator, tiny_model, pipeline_config):
    with pytest.raises(ValidationError):
        _init(tiny_model, pipeline_config, config={"train_batch_size": 8, "ray": {}})
    with pytest.raises(ValidationError):
        _init(tiny_model, pipeline_config, config={"pipeline": {"stages": 2}})


def test_unserializable_loss_fn_rejected(fake_coordinator, tiny_model, pipeline_config):
    import threading

    lock = threading.Lock()

    def loss_with_lock(outputs, labels):
        with lock:
            return outputs.sum()

    with pytest.raises(ValidationError):
        _init(tiny_model, pipeline_config, loss_fn=loss_with_lock)


def test_pipeline_config_type_enforced(fake_coordinator, tiny_model):
    with pytest.raises(ValidationError):
        _init(tiny_model, {"stages": 2})
