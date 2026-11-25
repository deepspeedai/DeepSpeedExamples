# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""The public return/property tree contains no Ray ObjectRef, actor handle or
other ray-typed object."""

import torch
from conftest import simple_loss_fn
from torch.utils.data import DataLoader

import ray_deepspeed_pipeline as rdsp


def _is_ray_module(module: str) -> bool:
    return module == "ray" or module.startswith("ray.")


def _assert_no_ray_types(obj, path="root", depth=0):
    assert not _is_ray_module(type(obj).__module__), f"{path} leaks a ray type: {type(obj)}"
    if depth >= 2 or isinstance(
        obj, (str, bytes, int, float, bool, type(None), torch.Tensor, DataLoader)
    ):
        return
    if isinstance(obj, (list, tuple, set)):
        for i, item in enumerate(obj):
            _assert_no_ray_types(item, f"{path}[{i}]", depth + 1)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_ray_types(v, f"{path}[{k!r}]", depth + 1)


def test_public_tree_has_no_ray_types(fake_coordinator, tiny_model, tiny_dataset, pipeline_config):
    result = rdsp.initialize(
        model=tiny_model,
        training_data=tiny_dataset,
        pipeline_config=pipeline_config,
        loss_fn=simple_loss_fn,
    )
    _assert_no_ray_types(result)

    engine = result[0]
    _assert_no_ray_types(engine.train_batch(), "train_batch()")
    _assert_no_ray_types(engine.eval_batch(), "eval_batch()")
    _assert_no_ray_types(engine.global_steps, "global_steps")
    # public (non-underscore, non-raising) attribute surface
    for name in ("train_batch", "eval_batch", "save_checkpoint", "load_checkpoint", "global_steps"):
        _assert_no_ray_types(getattr(engine, name), f"engine.{name}")
