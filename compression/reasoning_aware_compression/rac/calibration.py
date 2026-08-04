# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Calibration sets for one-shot pruning -- this is where RAC happens.

Layer-wise pruning minimises ``||W X - W' X||_F^2`` over a calibration
activation matrix ``X``. Three ways of building the token stream behind ``X``
are supported, matching the ablation in the RAC paper:

``c4``
    Generic web text (the SparseGPT/Wanda default). Nothing about the task or
    the model's own behaviour is represented.
``prompt``
    Task prompts only (e.g. OpenR1-Math-220k problems), chat-templated the way
    the model will see them at inference. Covers ``X^P`` from the paper.
``rac``
    Task prompts *concatenated with the dense model's own on-policy
    chain-of-thought*, collected by ``collect_traces.py``. Covers
    ``X^RAC = [X^P, X^D]``.

Teacher-forcing a sampled trace reproduces exactly the hidden states the model
computed while generating it, so packing traces into the calibration stream
recovers the decode-time activations of Algorithm 1 without re-running
generation during pruning.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import torch

CALIBRATION_SOURCES = ("c4", "prompt", "rac")

# Enough documents to fill a 1M-token budget with room to spare, so that the
# shuffle in :func:`pack_token_windows` has something to choose from.
_DEFAULT_TEXT_LIMIT = 20000


def chat_template_prompt(
    tokenizer,
    prompt: str,
    system_prompt: Optional[str] = None,
    use_chat_template: bool = True,
) -> str:
    """Render a raw prompt the way the model will receive it at inference."""
    if not use_chat_template or tokenizer.chat_template is None:
        return prompt

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_trace_texts(path: str, with_completion: bool = True) -> Iterator[str]:
    """Read the JSONL written by ``collect_traces.py``.

    Each line holds the chat-templated ``prompt`` and the model's
    ``completion``. ``with_completion=False`` yields the prompt-only variant of
    the *same* prompts, which is the apples-to-apples baseline for RAC.
    """
    trace_path = Path(path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file '{path}' not found. Run collect_traces.py first.")

    with trace_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row["prompt"]
            if with_completion:
                text = text + row["completion"]
            if text:
                yield text


def load_prompt_texts(
    dataset_name: str,
    prompt_column: str,
    tokenizer,
    split: str = "train",
    dataset_config: Optional[str] = None,
    system_prompt: Optional[str] = None,
    use_chat_template: bool = True,
    limit: int = _DEFAULT_TEXT_LIMIT,
) -> Iterator[str]:
    """Chat-templated task prompts from a Hugging Face dataset or a JSONL file."""
    for i, prompt in enumerate(_iter_raw_prompts(dataset_name, prompt_column, split,
                                                 dataset_config)):
        if i >= limit:
            break
        if prompt:
            yield chat_template_prompt(tokenizer, prompt, system_prompt, use_chat_template)


def load_c4_texts(
    dataset_name: str = "allenai/c4",
    dataset_config: str = "en",
    split: str = "train",
    limit: int = _DEFAULT_TEXT_LIMIT,
) -> Iterator[str]:
    """Stream raw web text -- the calibration set RAC is measured against."""
    from datasets import load_dataset

    stream = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    for i, row in enumerate(stream):
        if i >= limit:
            break
        text = row.get("text")
        if text:
            yield text


def pack_token_windows(
    texts: Iterable[str],
    tokenizer,
    nsamples: int,
    seqlen: int,
    seed: int = 0,
    shuffle_buffer: int = 4096,
) -> List[torch.Tensor]:
    """Tokenise ``texts`` and cut the stream into ``nsamples`` windows.

    Documents are separated by EOS and concatenated, then chunked into
    fixed-length windows. Fixed length keeps the causal mask and rotary
    embeddings identical across samples, which is what lets
    :mod:`rac.sequential` capture the block kwargs once, and it avoids padding
    tokens polluting the Hessian.

    Reasoning traces are typically several thousand tokens, so in ``rac`` mode
    most windows come from a single trace.
    """
    if nsamples <= 0 or seqlen <= 0:
        raise ValueError("nsamples and seqlen must both be positive.")

    rng = random.Random(seed)
    buffer: List[str] = []
    samples: List[torch.Tensor] = []
    stream: List[int] = []
    eos_id = tokenizer.eos_token_id

    def flush(docs: Sequence[str]) -> None:
        for doc in docs:
            ids = tokenizer(doc, add_special_tokens=False).input_ids
            stream.extend(ids)
            if eos_id is not None:
                stream.append(eos_id)
            while len(stream) >= seqlen and len(samples) < nsamples:
                samples.append(torch.tensor(stream[:seqlen], dtype=torch.long))
                del stream[:seqlen]
            if len(samples) >= nsamples:
                return

    # Shuffle within a bounded buffer so that a small token budget still draws
    # from across the corpus without materialising all of it.
    for text in texts:
        buffer.append(text)
        if len(buffer) >= shuffle_buffer:
            rng.shuffle(buffer)
            flush(buffer)
            buffer = []
            if len(samples) >= nsamples:
                break

    if len(samples) < nsamples and buffer:
        rng.shuffle(buffer)
        flush(buffer)

    if len(samples) < nsamples:
        raise ValueError(f"Calibration corpus yielded only {len(samples)} windows of {seqlen} "
                         f"tokens, {nsamples} requested. Collect more traces, lower --nsamples, "
                         "or lower --seqlen.")
    return samples


def build_calibration_samples(
    source: str,
    tokenizer,
    nsamples: int,
    seqlen: int,
    seed: int = 0,
    traces: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None,
    dataset_split: str = "train",
    prompt_column: str = "problem",
    system_prompt: Optional[str] = None,
    use_chat_template: bool = True,
) -> List[torch.Tensor]:
    """Build ``nsamples`` calibration windows of ``seqlen`` tokens.

    ``source="prompt"`` reads prompts from ``traces`` when given (so the
    prompt-only baseline sees exactly the prompts RAC saw) and falls back to
    ``dataset_name`` otherwise.
    """
    if source not in CALIBRATION_SOURCES:
        raise ValueError(f"Unknown calibration source '{source}'; "
                         f"expected one of {list(CALIBRATION_SOURCES)}.")

    if source == "rac":
        if not traces:
            raise ValueError("--calibration rac requires --traces; run collect_traces.py first.")
        texts: Iterable[str] = load_trace_texts(traces, with_completion=True)
    elif source == "prompt":
        if traces:
            texts = load_trace_texts(traces, with_completion=False)
        elif dataset_name:
            texts = load_prompt_texts(
                dataset_name,
                prompt_column,
                tokenizer,
                split=dataset_split,
                dataset_config=dataset_config,
                system_prompt=system_prompt,
                use_chat_template=use_chat_template,
            )
        else:
            raise ValueError("--calibration prompt requires either --traces or --dataset.")
    else:
        texts = load_c4_texts(
            dataset_name or "allenai/c4",
            dataset_config or "en",
            dataset_split,
        )

    return pack_token_windows(texts, tokenizer, nsamples, seqlen, seed=seed)


def _iter_raw_prompts(
    dataset_name: str,
    prompt_column: str,
    split: str,
    dataset_config: Optional[str],
) -> Iterator[str]:
    """Yield the raw prompt strings of a HF dataset or a local JSONL file."""
    if dataset_name.endswith(".jsonl") or dataset_name.endswith(".json"):
        with open(dataset_name) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)[prompt_column]
        return

    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    if prompt_column not in dataset.column_names:
        raise KeyError(f"Column '{prompt_column}' not in {dataset_name} "
                       f"(available: {dataset.column_names}). Set --prompt-column.")
    for row in dataset:
        yield row[prompt_column]
