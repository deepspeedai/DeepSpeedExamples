# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Catch direction, temperature, mask, cache and lazy-dependency regressions."""

import builtins
from dataclasses import replace

import pytest
import torch

from config import OPSDConfig
from losses import streamed_distillation_loss
from teacher import TeacherLogitCache
from vocab_parallel import VocabShardLayout


def local_layout(width):
    return VocabShardLayout(width, 0, width, (0,), ((0, width),), None)


@pytest.mark.parametrize("direction", ["forward_kl", "reverse_kl"])
@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("chunk", [1, 3, 12])
def test_streamed_local_kl_matches_independent_dense_oracle(direction, temperature, chunk):
    torch.manual_seed(710)
    student = torch.randn(2, 7, 11, requires_grad=True)
    reference = student.detach().double().requires_grad_()
    teacher = torch.randn(2, 7, 11)
    mask = torch.tensor([[1, 0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 0, 1]], dtype=torch.float32)
    layout = local_layout(11)
    cache = TeacherLogitCache.from_gpu_logits(teacher, store_dtype=torch.bfloat16, layout=layout)
    quantized = teacher.to(torch.bfloat16).double()
    logp = torch.log_softmax(reference / temperature, dim=-1)
    logq = torch.log_softmax(quantized / temperature, dim=-1)
    tokens = (logq.exp() * (logq - logp)).sum(-1) if direction == "forward_kl" else (logp.exp() * (logp - logq)).sum(-1)
    expected = (tokens * mask).sum() * temperature**2 / mask.sum()
    result = streamed_distillation_loss(student, lambda a, b: cache.chunk_to_device(a, b, student.device),
                                       mask, loss_type=direction, temperature=temperature, chunk_size=chunk,
                                       loss_backend="vocab_parallel", vocab_layout=layout)
    result.backward()
    expected.backward()
    torch.testing.assert_close(result.double(), expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(student.grad.double(), reference.grad, atol=1e-6, rtol=1e-5)


def test_chunks_without_response_tokens_are_skipped_and_backward_safe():
    student = torch.randn(1, 4, 9, requires_grad=True)
    fetched = []

    def fetch(start, end):
        fetched.append((start, end))
        return torch.zeros(1, end - start, 9)

    result = streamed_distillation_loss(student, fetch, torch.zeros(1, 4), chunk_size=3,
                                       loss_backend="vocab_parallel", vocab_layout=local_layout(9))
    result.backward()
    assert result.item() == 0
    assert fetched == []
    assert torch.count_nonzero(student.grad) == 0

    student.grad = None
    result = streamed_distillation_loss(student, fetch, torch.tensor([[0.0, 0.0, 0.0, 1.0]]), chunk_size=3,
                                       loss_backend="vocab_parallel", vocab_layout=local_layout(9))
    result.backward()
    assert fetched == [(3, 4)]
    assert torch.count_nonzero(student.grad[:, :3]) == 0


def test_cache_binds_actual_width_and_rejects_use_after_free():
    with pytest.raises(ValueError, match="width"):
        TeacherLogitCache(torch.zeros(1, 3, 8), layout=local_layout(9))
    cache = TeacherLogitCache(torch.zeros(1, 3, 9), layout=local_layout(9))
    cache.free()
    with pytest.raises(RuntimeError, match="freed"):
        cache.chunk_to_device(0, 1, torch.device("cpu"))


def test_dense_does_not_import_optional_kl_dependency(monkeypatch):
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "deepspeed.sequence.vocab_parallel_ops":
            raise ImportError("dependency absent")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    logits = torch.randn(1, 2, 5, requires_grad=True)
    fetch = lambda a, b: torch.zeros(1, b - a, 5)
    assert torch.isfinite(streamed_distillation_loss(logits, fetch, torch.ones(1, 2)))
    with pytest.raises(RuntimeError, match="PR #8791"):
        streamed_distillation_loss(logits, fetch, torch.ones(1, 2), loss_backend="vocab_parallel",
                                    vocab_layout=local_layout(5))


def test_old_config_defaults_dense_and_opt_in_rejects_unsupported_routes():
    cfg = OPSDConfig.from_dict({"student": {"model_name_or_path": "student"},
                               "teacher": {"model_name_or_path": "teacher"}})
    assert cfg.distillation.loss_backend == "dense"
    cfg.validate()
    cfg.distillation = replace(cfg.distillation, loss_backend="vocab_parallel", loss_type="jsd")
    with pytest.raises(ValueError, match="JSD"):
        cfg.validate()
    cfg.distillation.loss_type = "reverse_kl"
    cfg.rollout.temperature = 1.0
    with pytest.raises(ValueError, match="greedy"):
        cfg.validate()
    cfg.rollout.temperature = 0.0
    cfg.validate()
    for feature in ["use_graph_capture", "use_shared_prefill"]:
        setattr(cfg.rollout, feature, True)
        with pytest.raises(ValueError, match="eager"):
            cfg.validate()
        setattr(cfg.rollout, feature, False)


@pytest.mark.parametrize("implementation,backend", [
    ("upstream_dense", "dense"), ("dense_default", "dense"),
    ("gathered_reference", "vocab_parallel"), ("vocab_parallel", "vocab_parallel"),
])
def test_benchmark_requires_matching_backend_and_creates_output(tmp_path, implementation, backend):
    import json
    from benchmarks.bench_vocab_parallel_opsd import prepare_benchmark_inputs

    ds_path = tmp_path / "ds.json"
    ds_path.write_text(json.dumps({"tensor_parallel": {"autotp_size": 2}}))
    config_path = tmp_path / "opsd.json"
    config = {"deepspeed_config": str(ds_path), "distillation": {"loss_backend": backend}}
    config_path.write_text(json.dumps(config))
    output = tmp_path / "fresh" / "nested" / "results.jsonl"
    prepare_benchmark_inputs(implementation, config_path, output)
    assert output.is_file()
    output.write_text("existing result\n")
    prepare_benchmark_inputs(implementation, config_path, output)
    assert output.read_text() == "existing result\n"

    config["distillation"]["loss_backend"] = "dense" if backend == "vocab_parallel" else "vocab_parallel"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="requires distillation.loss_backend"):
        prepare_benchmark_inputs(implementation, config_path, output)


def test_benchmark_rejects_sharded_dense_heads_and_unwritable_output(tmp_path):
    import json
    from benchmarks.bench_vocab_parallel_opsd import prepare_benchmark_inputs

    ds_path = tmp_path / "ds.json"
    ds_path.write_text(json.dumps({"tensor_parallel": {"autotp_size": 2, "vocab_parallel_lm_head": True}}))
    config_path = tmp_path / "opsd.json"
    config_path.write_text(json.dumps({"deepspeed_config": str(ds_path)}))
    with pytest.raises(ValueError, match="full-vocabulary"):
        prepare_benchmark_inputs("dense_default", config_path, tmp_path / "results.jsonl")
    ds_path.write_text("{}")
    with pytest.raises(IsADirectoryError):
        prepare_benchmark_inputs("dense_default", config_path, tmp_path)
