# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""CPU-only tests for calibration-set construction.

Run with ``pytest`` from ``compression/reasoning_aware_compression``.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rac.calibration import (  # noqa: E402
    build_calibration_samples,
    chat_template_prompt,
    load_trace_texts,
    pack_token_windows,
)


class StubTokenizer:
    """Whitespace tokenizer, so the tests never touch the network."""

    eos_token_id = 0
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        ids = [(abs(hash(word)) % 1000) + 1 for word in text.split()]
        return type("Encoding", (), {"input_ids": ids})()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        rendered = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
        return rendered + "<|assistant|>" if add_generation_prompt else rendered


@pytest.fixture
def tokenizer():
    return StubTokenizer()


@pytest.fixture
def trace_file(tmp_path):
    path = tmp_path / "traces.jsonl"
    with path.open("w") as handle:
        for i in range(20):
            handle.write(
                json.dumps({
                    "prompt": " ".join(f"prompt{i}_{j}" for j in range(20)),
                    "completion": " ".join(f"cot{i}_{j}" for j in range(200)),
                    "prompt_tokens": 20,
                    "completion_tokens": 200,
                }) + "\n")
    return path


def test_pack_token_windows_shape(tokenizer):
    texts = [" ".join(f"tok{i}_{j}" for j in range(100)) for i in range(50)]
    samples = pack_token_windows(texts, tokenizer, nsamples=8, seqlen=64)

    assert len(samples) == 8
    for sample in samples:
        assert sample.shape == (64, )
        assert sample.dtype == torch.long


def test_pack_token_windows_is_deterministic(tokenizer):
    texts = [" ".join(f"tok{i}_{j}" for j in range(100)) for i in range(50)]
    a = pack_token_windows(texts, tokenizer, nsamples=4, seqlen=32, seed=7)
    b = pack_token_windows(texts, tokenizer, nsamples=4, seqlen=32, seed=7)

    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_pack_token_windows_rejects_short_corpus(tokenizer):
    with pytest.raises(ValueError, match="Calibration corpus yielded only"):
        pack_token_windows(["one two three"], tokenizer, nsamples=4, seqlen=64)


def test_rac_source_includes_cot_tokens(trace_file, tokenizer):
    """The whole method: RAC calibration must contain decode-time tokens."""
    prompt_only = list(load_trace_texts(str(trace_file), with_completion=False))
    with_cot = list(load_trace_texts(str(trace_file), with_completion=True))

    assert len(prompt_only) == len(with_cot) == 20
    for prompt, full in zip(prompt_only, with_cot):
        assert full.startswith(prompt)
        assert len(full) > len(prompt)
        assert "cot" in full and "cot" not in prompt


def test_build_calibration_samples_rac_vs_prompt(trace_file, tokenizer):
    # 20 traces x (20 prompt + 200 CoT) tokens is plenty for a 1024-token budget.
    rac = build_calibration_samples("rac", tokenizer, nsamples=16, seqlen=64,
                                    traces=str(trace_file))
    assert len(rac) == 16

    # The same traces hold 10x fewer prompt tokens, so the prompt-only baseline
    # runs out at the same budget -- the CoT is where the tokens are.
    with pytest.raises(ValueError, match="Calibration corpus yielded only"):
        build_calibration_samples("prompt", tokenizer, nsamples=16, seqlen=64,
                                  traces=str(trace_file))


def test_build_calibration_samples_validates_inputs(tokenizer):
    with pytest.raises(ValueError, match="requires --traces"):
        build_calibration_samples("rac", tokenizer, nsamples=1, seqlen=8)

    with pytest.raises(ValueError, match="requires either --traces or --dataset"):
        build_calibration_samples("prompt", tokenizer, nsamples=1, seqlen=8)

    with pytest.raises(ValueError, match="Unknown calibration source"):
        build_calibration_samples("wikitext", tokenizer, nsamples=1, seqlen=8)


def test_chat_template_prompt(tokenizer):
    assert chat_template_prompt(tokenizer, "2+2?", use_chat_template=False) == "2+2?"
    # No chat_template on the stub tokenizer means the raw prompt is returned.
    assert chat_template_prompt(tokenizer, "2+2?") == "2+2?"

    tokenizer.chat_template = "stub"
    rendered = chat_template_prompt(tokenizer, "2+2?", system_prompt="be brief")
    assert rendered == "<|system|>be brief<|user|>2+2?<|assistant|>"


def test_load_trace_texts_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run collect_traces.py first"):
        list(load_trace_texts(str(tmp_path / "nope.jsonl")))
