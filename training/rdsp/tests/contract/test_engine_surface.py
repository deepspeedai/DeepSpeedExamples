# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""RayPipelineEngine facade behavior."""

import pytest
import torch
from conftest import simple_loss_fn

import ray_deepspeed_pipeline as rdsp
from ray_deepspeed_pipeline.errors import UnsupportedEngineMethod, ValidationError


@pytest.fixture
def engine(fake_coordinator, tiny_model, tiny_dataset, pipeline_config):
    engine, _, _, _ = rdsp.initialize(
        model=tiny_model,
        training_data=tiny_dataset,
        pipeline_config=pipeline_config,
        loss_fn=simple_loss_fn,
    )
    return engine


def _assert_public_loss(loss):
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()
    assert loss.device.type == "cpu"
    assert loss.requires_grad is False
    assert not hasattr(loss, "result"), "futures must be resolved inside the facade"


def test_train_batch_resolves_future_to_detached_cpu_scalar(engine, fake_coordinator):
    loss = engine.train_batch()
    _assert_public_loss(loss)
    assert loss.item() == pytest.approx(0.5)
    assert fake_coordinator.train_calls == 1


def test_eval_batch_resolves_future_to_detached_cpu_scalar(engine, fake_coordinator):
    loss = engine.eval_batch()
    _assert_public_loss(loss)
    assert loss.item() == pytest.approx(0.25)


def test_global_steps_is_read_only_int(engine):
    assert engine.global_steps == 0
    engine.train_batch()
    assert engine.global_steps == 1
    with pytest.raises(AttributeError):
        engine.global_steps = 7


def test_unsupported_forward_backward_step(engine):
    for call in (
        lambda: engine(torch.zeros(1)),
        lambda: engine.forward(torch.zeros(1)),
        lambda: engine.backward(torch.zeros(())),
        lambda: engine.step(),
    ):
        with pytest.raises(UnsupportedEngineMethod) as e:
            call()
        assert "train_batch" in str(e.value), "the error must direct users to train_batch()"


def test_no_local_module_or_parameters(engine):
    with pytest.raises(UnsupportedEngineMethod):
        _ = engine.module
    with pytest.raises(UnsupportedEngineMethod):
        engine.parameters()


def test_not_a_deepspeed_engine_subclass(engine):
    assert all(cls.__name__ != "DeepSpeedEngine" for cls in type(engine).__mro__)


def test_engine_owned_loader_rejects_explicit_data_iter(engine):
    with pytest.raises(ValidationError):
        engine.train_batch(data_iter=iter([]))


def test_user_iterator_mode(fake_coordinator, tiny_model, tiny_dataset, pipeline_config):
    engine, _, dataloader, _ = rdsp.initialize(
        model=tiny_model,
        training_data=None,
        pipeline_config=pipeline_config,
        loss_fn=simple_loss_fn,
    )
    assert dataloader is None
    with pytest.raises(ValidationError):
        engine.train_batch()  # no data source at all
    loss = engine.train_batch(data_iter=iter(tiny_dataset))
    _assert_public_loss(loss)
