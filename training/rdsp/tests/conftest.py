# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import os
import sys

import pytest

# stage-local dispatch: a dead or failed neighbour surfaces within seconds in
# tests (production defaults are minutes)
os.environ.setdefault("RDSP_P2P_TIMEOUT_S", "15")
os.environ.setdefault("RDSP_ALIVE_TIMEOUT_S", "20")
import torch

# shared fixtures (ToyLM, stub engines) live in tests/unit; make them
# importable from the other test directories
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "unit"))

import ray_deepspeed_pipeline as rdsp
from ray_deepspeed_pipeline import api


class FakeFuture:
    """Stand-in for an internal future/reference the coordinator might return.
    The facade must resolve it; the public API must never surface it."""

    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class FakeCoordinator:
    def __init__(self):
        self.global_steps = 0
        self.train_calls = 0
        self.eval_calls = 0

    def train_batch(self, data_iter):
        self.train_calls += 1
        self.global_steps += 1
        return FakeFuture(0.5)

    def eval_batch(self, data_iter):
        self.eval_calls += 1
        return FakeFuture(0.25)

    def save_checkpoint(self, save_dir, tag=None, **kwargs):
        return FakeFuture(True)

    def load_checkpoint(self, load_dir, tag=None, **kwargs):
        return FakeFuture({"tag": tag})


@pytest.fixture
def fake_coordinator(monkeypatch):
    """Replace the internal coordinator factory (test-only injection point;
    never part of the public signature)."""
    coordinator = FakeCoordinator()
    monkeypatch.setattr(api, "_coordinator_factory", lambda **kwargs: coordinator)
    return coordinator


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))


@pytest.fixture
def tiny_dataset():
    # data contract: each entry is an (inputs, labels) pair
    return [(torch.randn(4), torch.tensor(0)) for _ in range(16)]


def simple_loss_fn(outputs, labels):
    return torch.nn.functional.cross_entropy(outputs, labels)


@pytest.fixture
def pipeline_config():
    return rdsp.PipelineConfig(stages=2, partition=rdsp.UniformTransformerBlocks())
