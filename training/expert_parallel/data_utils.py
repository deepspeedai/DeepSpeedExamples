"""Hugging Face text data loading and MoE config helpers for the AutoEP example."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import torch
from datasets import DownloadConfig, load_dataset
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, Llama4TextConfig, MixtralConfig, Qwen3_5MoeTextConfig


def qwen3_5_moe_layer_types(
    num_hidden_layers: int, *, full_attention_interval: int = 4
) -> list[str]:
    """HF Qwen3.5-MoE text stack requires at least one ``linear_attention`` layer (cache logic)."""
    return [
        "linear_attention"
        if bool((i + 1) % full_attention_interval)
        else "full_attention"
        for i in range(num_hidden_layers)
    ]


def build_qwen3_5_moe_text_config(
    *,
    num_hidden_layers: int | None = None,
) -> Qwen3_5MoeTextConfig:
    """Build the original Qwen3.5-MoE text config with an optional layer-count override."""
    kwargs: dict[str, Any] = {}
    if num_hidden_layers is not None:
        kwargs["num_hidden_layers"] = num_hidden_layers
    return Qwen3_5MoeTextConfig(**kwargs)


def build_llama4_text_config(
    *,
    num_hidden_layers: int | None = None,
) -> Llama4TextConfig:
    """Build the original Llama4 text config with an optional layer-count override."""
    kwargs: dict[str, Any] = {}
    if num_hidden_layers is not None:
        kwargs["num_hidden_layers"] = num_hidden_layers
    return Llama4TextConfig(**kwargs)


@dataclass
class CausalLmBatch:
    """One micro-batch for causal LM training (CPU tensors until the training loop moves them)."""

    input_ids: torch.Tensor  # [micro_batch_size, seq_len], dtype=torch.long
    attention_mask: torch.Tensor  # [micro_batch_size, seq_len], dtype=torch.long
    labels: torch.Tensor  # [micro_batch_size, seq_len], dtype=torch.long


def build_mixtral_config(
    num_layers: int | None = None,
) -> MixtralConfig:
    """Build the original Mixtral config with an optional layer-count override."""
    kwargs: dict[str, Any] = {}
    if num_layers is not None:
        kwargs["num_hidden_layers"] = num_layers
    return MixtralConfig(**kwargs)


def get_tokenizer(model_name: str, *, trust_remote_code: bool = True) -> Any:
    """Load tokenizer; pad token follows ds_verify_loss behavior."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(2)
    return tokenizer


def validate_tokenizer_vocab_size(
    tokenizer: Any,
    tokenizer_name: str,
    expected_vocab_size: int,
) -> dict[str, Any]:
    """Validate that tokenizer ids fit inside the model embedding table."""
    tokenizer_len = len(tokenizer)
    tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
    if tokenizer_len > expected_vocab_size:
        raise ValueError(
            f"Tokenizer {tokenizer_name!r} len(tokenizer)={tokenizer_len} "
            f"(vocab_size={tokenizer_vocab_size}) exceeds model "
            f"vocab_size={expected_vocab_size}. "
            "Pick a tokenizer whose ids fit within the model config."
        )
    return {
        "tokenizer_name": tokenizer_name,
        "tokenizer_len": tokenizer_len,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "model_vocab_size": expected_vocab_size,
        "exact_vocab_match": tokenizer_len == expected_vocab_size,
    }


def _hf_train_split_string(dataset_fraction: float) -> str:
    """Map fraction in (0, 1] to datasets split slice (matches ds_verify_loss)."""
    if dataset_fraction >= 1.0:
        return "train"
    percentage_int = int(dataset_fraction * 100)
    return f"train[:{percentage_int}%]"


def load_hf_text_dataset_rows(
    dataset_name: str,
    dataset_fraction: float,
    *,
    is_main_process: bool,
) -> tuple[Any, str]:
    """Load raw text rows from Hugging Face (same presets as ds_verify_loss)."""
    if not is_main_process:
        disable_progress_bar()

    split_str = _hf_train_split_string(dataset_fraction)
    dl_cfg = DownloadConfig(disable_tqdm=True)

    if is_main_process:
        print(f"Loading HF dataset: {dataset_name} split={split_str!r} ...")

    if dataset_name == "wikitext":
        dataset = load_dataset(
            "wikitext",
            "wikitext-103-raw-v1",
            split=split_str,
            download_config=dl_cfg,
        )
        text_column = "text"
    elif dataset_name == "openwebtext":
        if dataset_fraction >= 1.0:
            split_str = "train[:1%]"
        dataset = load_dataset(
            "openwebtext", split=split_str, download_config=dl_cfg
        )
        text_column = "text"
    elif dataset_name == "c4":
        if dataset_fraction >= 1.0:
            split_str = "train[:0.1%]"
        dataset = load_dataset(
            "c4", "en", split=split_str, download_config=dl_cfg
        )
        text_column = "text"
    elif dataset_name == "ag_news":
        dataset = load_dataset(
            "ag_news", split=split_str, download_config=dl_cfg
        )
        text_column = "text"
    else:
        try:
            dataset = load_dataset(
                dataset_name, split=split_str, download_config=dl_cfg
            )
            if "text" in dataset.column_names:
                text_column = "text"
            elif "content" in dataset.column_names:
                text_column = "content"
            elif "body" in dataset.column_names:
                text_column = "body"
            else:
                text_column = dataset.column_names[0]
                if is_main_process:
                    print(
                        f"Warning: using column {text_column!r}; "
                        f"columns={dataset.column_names}"
                    )
        except Exception as e:
            if is_main_process:
                print(f"Error loading {dataset_name!r}: {e}; falling back to wikitext.")
            dataset = load_dataset(
                "wikitext",
                "wikitext-103-raw-v1",
                split=split_str,
                download_config=dl_cfg,
            )
            text_column = "text"

    if is_main_process:
        print(f"HF dataset rows: {len(dataset)} (text column={text_column!r})")
    return dataset, text_column


def tokenize_hf_dataset(
    dataset: Any,
    text_column: str,
    tokenizer: Any,
    seq_len: int,
    *,
    is_main_process: bool,
) -> Any:
    """Tokenize text column to fixed length (padding=max_length), torch columns."""

    def has_text(example: dict[str, Any]) -> bool:
        value = example.get(text_column)
        return isinstance(value, str) and bool(value.strip())

    def tokenize_fn(examples: dict[str, list]) -> dict[str, list]:
        return tokenizer(
            examples[text_column],
            padding="max_length",
            max_length=seq_len,
            truncation=True,
        )

    if is_main_process:
        print("Filtering empty HF text rows...")
    dataset = dataset.filter(
        has_text,
        num_proc=1,
        keep_in_memory=True,
    )
    if len(dataset) == 0:
        raise ValueError(
            "HF dataset has no non-empty text rows; pick another dataset."
        )
    if is_main_process:
        print(f"Non-empty HF dataset rows: {len(dataset)}.")
        print("Tokenizing HF dataset...")
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=1,
        remove_columns=dataset.column_names,
        keep_in_memory=True,
    )
    tokenized.set_format(
        type="torch", columns=["input_ids", "attention_mask"]
    )
    if is_main_process:
        print(f"Tokenization complete: {len(tokenized)} rows.")
    if len(tokenized) == 0:
        raise ValueError(
            "Tokenized HF dataset is empty; increase dataset_percentage or pick another dataset."
        )
    return tokenized


class HFBatchGenerator:
    """Infinite iterator over a DataLoader; returns CausalLmBatch on CPU."""

    def __init__(
        self,
        dataloader: DataLoader,
        sampler: DistributedSampler,
    ) -> None:
        self.dataloader = dataloader
        self.sampler = sampler
        self._epoch = 0
        self._iter: Iterator | None = None

    def _next_raw_batch(self) -> dict[str, torch.Tensor]:
        if self._iter is None:
            self.sampler.set_epoch(self._epoch)
            self._iter = iter(self.dataloader)
        try:
            return next(self._iter)
        except StopIteration:
            self._epoch += 1
            self.sampler.set_epoch(self._epoch)
            self._iter = iter(self.dataloader)
            return next(self._iter)

    def get_batch(self, optimizer_step: int, accum_idx: int) -> CausalLmBatch:
        del optimizer_step, accum_idx  # sequential consumption (like an infinite epoch)
        batch = self._next_raw_batch()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = input_ids.clone()
        labels = labels.masked_fill(attention_mask == 0, -100)
        return CausalLmBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )


def build_hf_batch_generator(
    *,
    dataset_name: str,
    dataset_percentage: float,
    tokenizer_name: str,
    expected_vocab_size: int,
    seq_len: int,
    micro_batch_size: int,
    dp_world_size: int,
    dp_rank: int,
    seed: int,
    rank: int,
    hf_num_dataloader_workers: int = 0,
) -> HFBatchGenerator:
    """Load HF text data, tokenize, and build a per-DP-rank batch generator."""
    is_main = rank == 0
    if dataset_percentage <= 0:
        raise ValueError("dataset_percentage must be positive")
    if dataset_percentage < 1.0:
        raise ValueError(
            "dataset_percentage must be at least 1.0 because Hugging Face "
            "split slicing uses whole percentages."
        )
    fraction = min(dataset_percentage / 100.0, 1.0)

    tokenizer = get_tokenizer(tokenizer_name, trust_remote_code=True)
    validate_tokenizer_vocab_size(tokenizer, tokenizer_name, expected_vocab_size)
    raw, text_col = load_hf_text_dataset_rows(
        dataset_name, fraction, is_main_process=is_main
    )
    tokenized = tokenize_hf_dataset(
        raw,
        text_col,
        tokenizer,
        seq_len,
        is_main_process=is_main,
    )
    sampler = DistributedSampler(
        tokenized,
        num_replicas=dp_world_size,
        rank=dp_rank,
        shuffle=True,
        seed=seed,
    )
    loader = DataLoader(
        tokenized,
        batch_size=micro_batch_size,
        sampler=sampler,
        num_workers=hf_num_dataloader_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return HFBatchGenerator(loader, sampler)
