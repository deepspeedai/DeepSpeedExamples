# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""CPU-only tests for the prompt dataset and the left-padded collator.

A tiny stand-in tokenizer (no HF download needed) is enough to exercise the
collator's left-padding layout, truncation, pad-id fallback, and the
dataset's JSONL loading.
"""

import pytest
import torch

from data import LeftPaddedPromptCollator, PromptDataset


class _FakeTokenizer:
    """Minimal HF-tokenizer stand-in: exact string -> id list, with truncation.

    No ``apply_chat_template`` on purpose, so :class:`PromptDataset` returns the
    raw prompt text (lets us assert on it directly).
    """

    def __init__(self, vocab, pad_token_id=0, eos_token_id=1):
        self._vocab = dict(vocab)
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def __call__(self, text, add_special_tokens=False, truncation=True,
                 max_length=None, return_tensors="pt", **kwargs):
        ids = list(self._vocab[text])
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def test_collator_left_pads_real_tokens_to_right_edge():
    tok = _FakeTokenizer({"aa": [10, 11], "bbbb": [20, 21, 22, 23], "ccc": [30, 31, 32]})
    col = LeftPaddedPromptCollator(tok, max_prompt_length=16)
    out = col(["aa", "bbbb", "ccc"])
    ids, mask = out["prompt_ids"], out["prompt_attention_mask"]
    assert ids.shape == (3, 4)
    assert mask.shape == (3, 4)
    pad = tok.pad_token_id
    # "aa" (len 2): two pads on the left, real tokens at the right edge
    assert ids[0].tolist() == [pad, pad, 10, 11]
    assert mask[0].tolist() == [0, 0, 1, 1]
    # "bbbb" (len 4): the longest, no padding
    assert ids[1].tolist() == [20, 21, 22, 23]
    assert mask[1].tolist() == [1, 1, 1, 1]
    # "ccc" (len 3): one pad on the left
    assert ids[2].tolist() == [pad, 30, 31, 32]
    assert mask[2].tolist() == [0, 1, 1, 1]


def test_collator_truncates_to_max_prompt_length():
    tok = _FakeTokenizer({"long": [1, 2, 3, 4, 5]})
    col = LeftPaddedPromptCollator(tok, max_prompt_length=3)
    out = col(["long"])
    assert out["prompt_ids"].shape == (1, 3)
    assert out["prompt_ids"][0].tolist() == [1, 2, 3]
    assert out["prompt_attention_mask"][0].tolist() == [1, 1, 1]


def test_collator_falls_back_to_eos_when_no_pad_id():
    tok = _FakeTokenizer({"ab": [10, 11], "abcd": [20, 21, 22, 23]},
                         pad_token_id=None, eos_token_id=99)
    col = LeftPaddedPromptCollator(tok, max_prompt_length=16)
    out = col(["ab", "abcd"])
    # "ab" is left-padded to len 4 using the eos id (99)
    assert out["prompt_ids"][0].tolist() == [99, 99, 10, 11]
    assert out["prompt_attention_mask"][0].tolist() == [0, 0, 1, 1]


def test_collator_raises_without_pad_and_eos():
    tok = _FakeTokenizer({"ab": [10, 11]}, pad_token_id=None, eos_token_id=None)
    with pytest.raises(ValueError, match="neither pad_token_id nor eos_token_id"):
        LeftPaddedPromptCollator(tok, max_prompt_length=16)


def test_prompt_dataset_reads_jsonl(tmp_path):
    p = tmp_path / "prompts.jsonl"
    p.write_text('{"prompt": "hello"}\n\n{"prompt": "world"}\n')
    ds = PromptDataset(str(p), _FakeTokenizer({}), max_prompt_length=8)
    assert len(ds) == 2
    assert ds[0] == "hello"
    assert ds[1] == "world"


def test_prompt_dataset_missing_field_raises(tmp_path):
    p = tmp_path / "prompts.jsonl"
    p.write_text('{"text": "no prompt field here"}\n')
    ds = PromptDataset(str(p), _FakeTokenizer({}), max_prompt_length=8)
    with pytest.raises(KeyError, match="missing field 'prompt'"):
        ds[0]
