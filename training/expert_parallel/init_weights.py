"""Init-weights artifact helpers for AutoEP parity workflows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from typing import Any

import torch
from safetensors.torch import load_file, save_file

CURRENT_INIT_SCHEMA_VERSION = 1


def sha256_file(path: str) -> str:
    """Return SHA-256 hex digest of file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_config_fingerprint(model_config: object) -> str:
    """Build a stable fingerprint from shape-defining model config fields."""
    num_experts = getattr(model_config, "num_local_experts", None)
    if num_experts is None:
        num_experts = getattr(model_config, "num_experts", None)
    intermediate = getattr(model_config, "intermediate_size", None)
    if intermediate is None:
        intermediate = getattr(model_config, "moe_intermediate_size", None)
    fields = {
        "num_hidden_layers": getattr(model_config, "num_hidden_layers", None),
        "hidden_size": getattr(model_config, "hidden_size"),
        "intermediate_size": intermediate,
        "num_attention_heads": getattr(model_config, "num_attention_heads"),
        "num_key_value_heads": getattr(model_config, "num_key_value_heads"),
        "num_experts": num_experts,
        "num_experts_per_tok": getattr(model_config, "num_experts_per_tok"),
        "vocab_size": getattr(model_config, "vocab_size"),
        "max_position_embeddings": getattr(model_config, "max_position_embeddings"),
    }
    payload = json.dumps(fields, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _meta_path(weights_path: str) -> str:
    stem, _ = os.path.splitext(weights_path)
    return f"{stem}_meta.json"


def _ensure_safetensors_path(path: str) -> None:
    if not path.endswith(".safetensors"):
        raise ValueError(
            f"Init weights path must end with '.safetensors': {path}"
        )


def _fsync_dir(path: str) -> None:
    dir_fd = os.open(os.path.dirname(path) or ".", os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path)


def _model_state_dict_cpu_fp32(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    converted: dict[str, torch.Tensor] = {}
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].detach().cpu().contiguous()
        if torch.is_floating_point(tensor):
            tensor = tensor.to(torch.float32)
        converted[key] = tensor
    return converted


def _transformers_version() -> str:
    try:
        return importlib.metadata.version("transformers")
    except Exception:
        return "unknown"


def save_init_weights_artifact(
    path: str,
    model: torch.nn.Module,
    *,
    args: argparse.Namespace,
    model_config: object,
    rank: int,
) -> dict[str, object]:
    """Save init weights safetensors + sidecar metadata and return context."""
    _ensure_safetensors_path(path)

    out_path = os.path.abspath(path)
    meta_path = _meta_path(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    state_dict = _model_state_dict_cpu_fp32(model)
    tmp_weights = out_path + ".tmp"
    save_file(state_dict, tmp_weights)
    with open(tmp_weights, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp_weights, out_path)
    _fsync_dir(out_path)

    weights_sha256 = sha256_file(out_path)
    config_fingerprint = build_config_fingerprint(model_config)
    metadata = {
        "schema_version": CURRENT_INIT_SCHEMA_VERSION,
        "seed": int(getattr(args, "seed")),
        "num_layers": int(getattr(args, "num_layers")),
        "model_class": model.__class__.__name__,
        "config_fingerprint": config_fingerprint,
        "torch_version": torch.__version__,
        "transformers_version": _transformers_version(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "saved_by_rank": int(rank),
    }
    _write_json_atomic(meta_path, metadata)

    return {
        "init_weights_path": out_path,
        "init_weights_sha256": weights_sha256,
        "init_weights_loaded": False,
        "init_weights_schema_version": CURRENT_INIT_SCHEMA_VERSION,
    }


def load_init_weights_artifact(
    path: str,
    model: torch.nn.Module,
    *,
    args: argparse.Namespace,
    model_config: object,
) -> dict[str, object]:
    """Load and validate init weights artifact, then return metadata context."""
    _ensure_safetensors_path(path)
    in_path = os.path.abspath(path)
    meta_path = _meta_path(in_path)

    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Init weights file does not exist: {in_path}")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Init weights metadata file does not exist: {meta_path}")

    with open(meta_path) as f:
        metadata = json.load(f)

    schema_version = int(metadata.get("schema_version", 0))
    if schema_version < 1:
        raise ValueError(f"Invalid init-weights schema version: {schema_version}")
    if schema_version > CURRENT_INIT_SCHEMA_VERSION:
        raise ValueError(
            "Schema version "
            f"{schema_version} is not supported. Please upgrade your tools."
        )

    expected_layers = int(getattr(args, "num_layers"))
    artifact_layers = int(metadata.get("num_layers", -1))
    if artifact_layers != expected_layers:
        raise ValueError(
            "Init weights num_layers mismatch: "
            f"artifact={artifact_layers}, run={expected_layers}"
        )

    expected_fingerprint = build_config_fingerprint(model_config)
    artifact_fingerprint = metadata.get("config_fingerprint")
    if artifact_fingerprint != expected_fingerprint:
        raise ValueError(
            "Init weights config fingerprint mismatch: "
            "artifact does not match current model configuration."
        )

    state_dict = load_file(in_path, device="cpu")
    model.load_state_dict(state_dict, strict=True)

    return {
        "init_weights_path": in_path,
        "init_weights_sha256": sha256_file(in_path),
        "init_weights_loaded": True,
        "init_weights_schema_version": schema_version,
    }
