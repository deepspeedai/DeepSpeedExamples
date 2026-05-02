"""MoE causal LM training entry point (AutoEP or ZeRO-3 leaf).

Runs a randomly initialized Mixtral-, Llama4-, or Qwen3.5-MoE-style model in either
AutoEP+ZeRO-1 mode or HF-native+ZeRO-3 leaf-module mode (no expert-parallel
routing in the latter), collecting per-step metrics when configured.

Training data is loaded from Hugging Face (tokenization aligned with ``ds_verify_loss``).

Launch via deepspeed launcher (built-in DeepSpeed JSON is derived from ``--mode`` and ``--model``;
optional ``--deepspeed_config`` merges a JSON file and still syncs AutoEP ``preset_model`` /
ZeRO-3 ``leaf_module`` to the model architecture):
    deepspeed --num_gpus 8 train.py --mode autoep
    deepspeed --num_gpus 8 train.py --mode zero3_leaf
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from typing import Any, NamedTuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, Llama4ForCausalLM, Qwen3_5MoeForCausalLM

import deepspeed

from data_utils import (
    build_hf_batch_generator,
    build_llama4_text_config,
    build_mixtral_config,
    build_qwen3_5_moe_text_config,
    get_tokenizer,
    validate_tokenizer_vocab_size,
)
from init_weights import load_init_weights_artifact, save_init_weights_artifact
from metrics import MetricsLogger, reduce_loss, reduce_max, write_run_metadata
from validation import (
    collect_run_metadata,
    validate_autoep_engine,
    validate_zero3_leaf_engine,
)

logger = logging.getLogger(__name__)

# Public model presets describe original Hugging Face model-family layouts.
# DeepSpeed's ``preset_model`` is structural injection metadata and must not
# encode synthetic sizes or layer counts.
MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "mixtral_8x7b": {
        "architecture": "mixtral",
        "display_name": "Mixtral 8x7B",
        "default_tokenizer_name": "mistralai/Mixtral-8x7B-v0.1",
    },
    "qwen3_5": {
        "architecture": "qwen3_5_moe",
        "display_name": "Qwen3.5 MoE",
        "default_tokenizer_name": "Qwen/Qwen3-0.6B",
    },
    "llama4": {
        "architecture": "llama4",
        "display_name": "Llama4 Scout",
        "default_tokenizer_name": "meta-llama/Llama-4-Scout-17B-16E",
    },
}

# DeepSpeed AutoEP structural preset id (must match HF MoE layout for this example).
DEEPSPEED_AUTOEP_PRESET_ID: dict[str, str] = {
    "llama4": "llama4",
    "mixtral": "mixtral",
    "qwen3_5_moe": "qwen3_5_moe",
}

# ZeRO-3 leaf module: full class path for the HF SparseMoeBlock used by each architecture.
DEEPSPEED_LEAF_MOE_BLOCK_CLASS: dict[str, str] = {
    "llama4": (
        "transformers.models.llama4.modeling_llama4.Llama4TextMoe"
    ),
    "mixtral": (
        "transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock"
    ),
    "qwen3_5_moe": (
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe."
        "Qwen3_5MoeSparseMoeBlock"
    ),
}


def default_autoep_parallel_size(architecture: str) -> int:
    """Default ``expert_parallel.autoep_size`` when not set in JSON or ``--autoep_size``."""
    if architecture == "qwen3_5_moe":
        return 8
    if architecture in {"llama4", "mixtral"}:
        return 4
    raise ValueError(f"Unknown architecture for autoep_size default: {architecture!r}")


def default_autoep_parallel_size_for_model(model: str) -> int:
    """Same as ``default_autoep_parallel_size`` for a ``--model`` preset name."""
    if model not in MODEL_PRESETS:
        raise ValueError(f"Unknown model preset: {model!r}")
    return default_autoep_parallel_size(MODEL_PRESETS[model]["architecture"])


def parse_boolish(value: str) -> bool:
    """Parse common CLI spellings for booleans."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "expected one of: true/false, yes/no, 1/0, on/off"
    )


def resolve_output_router_logits(args: argparse.Namespace) -> bool:
    """Resolve new ``--output_router_logits`` alias while preserving legacy behavior."""
    if args.output_router_logits is not None:
        return bool(args.output_router_logits)
    return args.include_router_aux_loss == "on"


def _ds_scheduler_dict() -> dict[str, Any]:
    return {
        "type": "WarmupCosineLR",
        "params": {
            "total_num_steps": 1000,
            "warmup_min_ratio": 0,
            "warmup_num_steps": 100,
            "cos_min_ratio": 0.001,
            "warmup_type": "linear",
        },
    }


def _ds_optimizer_dict() -> dict[str, Any]:
    return {
        "type": "AdamW",
        "params": {"lr": 1e-4},
    }


def build_default_deepspeed_config(mode: str, architecture: str) -> dict[str, Any]:
    """Full DeepSpeed config dict aligned with ``--mode`` and model ``architecture``."""
    if architecture not in DEEPSPEED_AUTOEP_PRESET_ID:
        raise ValueError(f"No DeepSpeed mapping for architecture: {architecture!r}")
    base: dict[str, Any] = {
        "bf16": {"enabled": True},
        "optimizer": _ds_optimizer_dict(),
        "scheduler": _ds_scheduler_dict(),
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 10,
    }
    if mode == "autoep":
        base["zero_optimization"] = {"stage": 1}
        base["expert_parallel"] = {
            "enabled": True,
            "autoep_size": default_autoep_parallel_size(architecture),
            "preset_model": DEEPSPEED_AUTOEP_PRESET_ID[architecture],
            "load_balance_coeff": None,
        }
    elif mode == "zero3_leaf":
        base["zero_optimization"] = {
            "stage": 3,
            "stage3_param_persistence_threshold": 1e5,
            "leaf_module": {
                "classes": [DEEPSPEED_LEAF_MOE_BLOCK_CLASS[architecture]],
            },
        }
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    return base


def apply_architecture_to_ds_config(
    ds_config: dict[str, Any], mode: str, architecture: str
) -> None:
    """Force AutoEP / ZeRO-3 leaf fields to match ``architecture`` (mutates ``ds_config``)."""
    if architecture not in DEEPSPEED_AUTOEP_PRESET_ID:
        raise ValueError(f"No DeepSpeed mapping for architecture: {architecture!r}")
    if mode == "autoep":
        ep = ds_config.setdefault("expert_parallel", {})
        ep["enabled"] = True
        ep["preset_model"] = DEEPSPEED_AUTOEP_PRESET_ID[architecture]
        if "autoep_size" not in ep:
            ep["autoep_size"] = default_autoep_parallel_size(architecture)
        if "load_balance_coeff" not in ep:
            ep["load_balance_coeff"] = None
    elif mode == "zero3_leaf":
        zo = ds_config.setdefault("zero_optimization", {})
        lm = zo.setdefault("leaf_module", {})
        lm["classes"] = [DEEPSPEED_LEAF_MOE_BLOCK_CLASS[architecture]]
    else:
        raise ValueError(f"Unknown mode: {mode!r}")


def resolve_deepspeed_config(
    config_path: str | None, mode: str, architecture: str
) -> dict[str, Any]:
    """Load JSON from ``config_path`` or use built-in defaults; always sync architecture fields."""
    if config_path is not None:
        ds_config = load_ds_config(config_path)
        apply_architecture_to_ds_config(ds_config, mode, architecture)
        return ds_config
    return build_default_deepspeed_config(mode, architecture)


class ResolvedModelPreset(NamedTuple):
    architecture: str
    display_name: str
    default_tokenizer_name: str
    num_layers_overridden: bool


def build_original_model_config(
    architecture: str,
    *,
    num_layers: int | None,
    output_router_logits: bool,
) -> Any:
    """Build an original HF model-family config, overriding only layer count."""
    if architecture == "mixtral":
        return build_mixtral_config(
            num_layers=num_layers,
            output_router_logits=output_router_logits,
        )
    if architecture == "qwen3_5_moe":
        return build_qwen3_5_moe_text_config(
            num_hidden_layers=num_layers,
            output_router_logits=output_router_logits,
        )
    if architecture == "llama4":
        return build_llama4_text_config(
            num_hidden_layers=num_layers,
            output_router_logits=output_router_logits,
        )
    raise ValueError(f"Unsupported architecture: {architecture!r}")


def num_experts_for_config(architecture: str, model_config: Any) -> int:
    """Return the routed expert count for a built HF config."""
    if architecture in {"mixtral", "llama4"}:
        return int(model_config.num_local_experts)
    if architecture == "qwen3_5_moe":
        return int(model_config.num_experts)
    raise ValueError(f"Unsupported architecture: {architecture!r}")


def validate_autoep_size(
    *,
    architecture: str,
    autoep_size: int,
    num_experts: int,
    world_size: int,
) -> None:
    """Fail fast on invalid AutoEP topology before DeepSpeed engine init."""
    valid_expert_divisors = [d for d in range(1, num_experts + 1) if num_experts % d == 0]
    valid_world_divisors = [d for d in range(1, world_size + 1) if world_size % d == 0]
    valid_sizes = [d for d in valid_expert_divisors if d in valid_world_divisors]
    if autoep_size not in valid_sizes:
        raise ValueError(
            "Invalid AutoEP size for "
            f"architecture={architecture!r}: autoep_size={autoep_size}, "
            f"num_experts={num_experts}, world_size={world_size}. "
            f"Valid values that divide both num_experts and world_size: {valid_sizes}"
        )


def validate_autoep_structural_preset_available(preset_id: str) -> None:
    """Fail early when the imported DeepSpeed does not expose the needed AutoEP preset."""
    try:
        from deepspeed.module_inject.auto_ep_config import PRESET_MODELS
    except ImportError as exc:
        raise ValueError(
            "Imported DeepSpeed does not expose AutoEP preset metadata; "
            "install an AutoEP-enabled DeepSpeed build."
        ) from exc
    if preset_id not in PRESET_MODELS:
        raise ValueError(
            f"Imported DeepSpeed does not provide AutoEP preset_model={preset_id!r}. "
            f"Available presets: {sorted(PRESET_MODELS)}"
        )


def apply_model_preset(args: argparse.Namespace) -> ResolvedModelPreset:
    """Resolve ``--model`` into architecture tag and builder kwargs."""
    if args.model not in MODEL_PRESETS:
        raise ValueError(f"Unknown model preset: {args.model!r}")
    preset = MODEL_PRESETS[args.model]
    architecture = preset["architecture"]
    num_layers_overridden = args.num_layers is not None
    if args.num_layers is None:
        original_config = build_original_model_config(
            architecture,
            num_layers=None,
            output_router_logits=False,
        )
        args.num_layers = int(original_config.num_hidden_layers)
    if args.tokenizer_name is None:
        args.tokenizer_name = preset["default_tokenizer_name"]
    args.num_layers_overridden = num_layers_overridden
    return ResolvedModelPreset(
        architecture=architecture,
        display_name=preset["display_name"],
        default_tokenizer_name=preset["default_tokenizer_name"],
        num_layers_overridden=num_layers_overridden,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MoE causal LM training (AutoEP or ZeRO-3 leaf; original "
            "Qwen3.5-MoE, Llama4, or Mixtral presets)"
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="autoep",
        choices=["autoep", "zero3_leaf"],
        help="Training mode (default: autoep)",
    )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default=None,
        help=(
            "Optional path to DeepSpeed JSON; merged then AutoEP preset_model / "
            "ZeRO-3 leaf_module.classes are forced to match --model. "
            "If omitted, a built-in config is used."
        ),
    )
    parser.add_argument("--steps", type=int, default=50, help="Total optimizer steps")
    parser.add_argument(
        "--warmup_steps", type=int, default=5, help="Warmup steps before measurement"
    )
    parser.add_argument(
        "--log_interval", type=int, default=1, help="Log every N optimizer steps"
    )
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument(
        "--micro_batch_size", type=int, default=2, help="Micro batch size per GPU"
    )
    parser.add_argument(
        "--grad_accum", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--target_global_tokens_per_update",
        type=int,
        default=None,
        help="Target global tokens per optimizer update; derives grad_accum per mode",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3_5",
        choices=sorted(MODEL_PRESETS.keys()),
        help=(
            "Original model-family preset. Public presets do not encode "
            "synthetic sizes; use --num_layers to override only depth."
        ),
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="Override only the HF config layer-count field.",
    )
    parser.add_argument(
        "--autoep_size",
        type=int,
        default=None,
        help="Override autoep_size (AutoEP mode only)",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=str,
        choices=["on", "off"],
        default="off",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--include_router_aux_loss",
        type=str,
        choices=["on", "off"],
        default="off",
        help="Include router auxiliary loss",
    )
    parser.add_argument(
        "--output_router_logits",
        "--output-router-logits",
        type=parse_boolish,
        default=None,
        help=(
            "Enable/disable model config output_router_logits. If omitted, "
            "the legacy --include_router_aux_loss setting is used."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable torch.use_deterministic_algorithms",
    )
    parser.add_argument(
        "--allow_untested_versions",
        action="store_true",
        help="Bypass version compatibility gate",
    )
    parser.add_argument(
        "--metrics_out", type=str, default=None, help="CSV output path"
    )
    parser.add_argument(
        "--run_metadata_out", type=str, default=None, help="Metadata JSON output path"
    )
    parser.add_argument(
        "--save_checkpoint",
        type=str,
        default=None,
        help="Save checkpoint to this directory after training",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="Load checkpoint from this directory before training",
    )
    parser.add_argument(
        "--save_init_weights",
        type=str,
        default=None,
        help="Save pre-DeepSpeed init weights artifact (.safetensors)",
    )
    parser.add_argument(
        "--load_init_weights",
        type=str,
        default=None,
        help="Load pre-DeepSpeed init weights artifact (.safetensors)",
    )
    parser.add_argument(
        "--init_weights_only",
        action="store_true",
        help="Save init weights artifact and exit before DeepSpeed initialization",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank passed by deepspeed launcher",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="HF dataset preset or hub id (see ds_verify_loss)",
    )
    parser.add_argument(
        "--dataset_percentage",
        type=float,
        default=10.0,
        help="Percent of train split to use (e.g. 10.0 = ten percent)",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help=(
            "HF tokenizer id; tokenizer ids must fit within model vocab_size "
            "(default: from --model preset)"
        ),
    )
    parser.add_argument(
        "--hf_num_dataloader_workers",
        type=int,
        default=0,
        help="DataLoader worker processes (0 is safest for distributed)",
    )
    args = parser.parse_args()
    validate_init_weight_args(args, parser)
    return args


def validate_init_weight_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """Validate init-weights/checkpoint argument combinations."""
    if args.init_weights_only and args.save_init_weights is None:
        parser.error("--init_weights_only requires --save_init_weights.")

    if args.init_weights_only and args.load_checkpoint is not None:
        parser.error("--init_weights_only is incompatible with --load_checkpoint.")

    if args.init_weights_only and args.save_checkpoint is not None:
        parser.error("--init_weights_only is incompatible with --save_checkpoint.")

    if args.load_init_weights is not None and args.load_checkpoint is not None:
        parser.error("--load_init_weights is incompatible with --load_checkpoint.")

    if args.save_init_weights is not None and args.load_init_weights is not None:
        parser.error(
            "--save_init_weights and --load_init_weights cannot be used together."
        )

    if args.save_init_weights is not None and not args.save_init_weights.endswith(
        ".safetensors"
    ):
        parser.error("--save_init_weights path must end with '.safetensors'.")

    if args.load_init_weights is not None:
        if not args.load_init_weights.endswith(".safetensors"):
            parser.error("--load_init_weights path must end with '.safetensors'.")
        if not os.path.isfile(args.load_init_weights):
            parser.error(
                f"--load_init_weights file does not exist: {args.load_init_weights}"
            )


def load_ds_config(config_path: str) -> dict:
    """Load and return DeepSpeed config as a dict."""
    with open(config_path) as f:
        return json.load(f)


def main():
    args = parse_args()
    resolved_preset = apply_model_preset(args)

    # Set defaults for output paths
    if args.metrics_out is None:
        args.metrics_out = f"metrics_{args.mode}.csv"
    if args.run_metadata_out is None:
        args.run_metadata_out = f"run_metadata_{args.mode}.json"

    if args.init_weights_only:
        rank = 0
        world_size = 1
    else:
        # deepspeed.initialize handles distributed setup, but we need rank for logging
        deepspeed.init_distributed()
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if torch.cuda.is_available():
            local_rank_env = int(os.environ.get("LOCAL_RANK", args.local_rank))
            if local_rank_env >= 0:
                torch.cuda.set_device(local_rank_env)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"[rank {rank}] %(levelname)s: %(message)s",
    )

    # Set seeds BEFORE model construction
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load or build DS config (always align AutoEP / ZeRO-3 leaf with --model)
    ds_config = resolve_deepspeed_config(
        args.deepspeed_config, args.mode, resolved_preset.architecture
    )
    if rank == 0:
        if args.deepspeed_config:
            logger.info(
                "DeepSpeed config from %s (architecture fields synced to %s).",
                args.deepspeed_config,
                resolved_preset.architecture,
            )
        else:
            logger.info(
                "DeepSpeed config built-in for mode=%s, architecture=%s.",
                args.mode,
                resolved_preset.architecture,
            )

    # Validate precision
    bf16_enabled = ds_config.get("bf16", {}).get("enabled", False)
    fp16_enabled = ds_config.get("fp16", {}).get("enabled", False)
    if not bf16_enabled and not fp16_enabled:
        logger.error(
            "Neither bf16 nor fp16 is enabled. FP32-only is not supported."
        )
        sys.exit(2)
    if fp16_enabled and not bf16_enabled:
        logger.warning(
            "fp16 is enabled but bf16 is preferred for Hopper grouped GEMM fast-path."
        )

    # Override batch config from CLI args
    ds_config["train_micro_batch_size_per_gpu"] = args.micro_batch_size
    ds_config["gradient_accumulation_steps"] = args.grad_accum
    # Remove train_batch_size if present; let DeepSpeed derive it
    ds_config.pop("train_batch_size", None)

    # Override autoep_size if provided
    if args.mode == "autoep" and args.autoep_size is not None:
        if "expert_parallel" not in ds_config:
            ds_config["expert_parallel"] = {}
        ds_config["expert_parallel"]["autoep_size"] = args.autoep_size

    # Read autoep_size from config for validation
    autoep_size = 1
    if args.mode == "autoep":
        autoep_size = ds_config.get("expert_parallel", {}).get("autoep_size", 1)
        if autoep_size == 1:
            logger.warning(
                "autoep_size=1: EP communication is bypassed (degenerate case). "
                "Set autoep_size >= 2 to test expert parallelism."
            )

    # Derive grad_accum from target_global_tokens_per_update if provided
    if args.target_global_tokens_per_update is not None:
        if args.mode == "autoep":
            dp_ws = world_size // autoep_size
        else:
            dp_ws = world_size
        tokens_per_microstep = args.seq_len * args.micro_batch_size * dp_ws
        if tokens_per_microstep == 0:
            logger.error("tokens_per_microstep is 0; check seq_len, micro_batch_size.")
            sys.exit(2)
        derived_ga = args.target_global_tokens_per_update / tokens_per_microstep
        if derived_ga != int(derived_ga) or derived_ga < 1:
            logger.error(
                f"target_global_tokens_per_update={args.target_global_tokens_per_update} "
                f"is not evenly divisible by tokens_per_microstep={tokens_per_microstep}. "
                f"Derived grad_accum={derived_ga} is not a positive integer."
            )
            sys.exit(2)
        args.grad_accum = int(derived_ga)
        ds_config["gradient_accumulation_steps"] = args.grad_accum
        if rank == 0:
            logger.info(
                f"Derived grad_accum={args.grad_accum} from "
                f"target_global_tokens_per_update={args.target_global_tokens_per_update}"
            )

    # Read load_balance_coeff for validation
    load_balance_coeff = None
    if args.mode == "autoep":
        load_balance_coeff = ds_config.get("expert_parallel", {}).get(
            "load_balance_coeff", None
        )

    # Build model config
    output_router_logits = resolve_output_router_logits(args)
    args.resolved_output_router_logits = output_router_logits
    model_config = build_original_model_config(
        resolved_preset.architecture,
        num_layers=args.num_layers,
        output_router_logits=output_router_logits,
    )
    num_experts = num_experts_for_config(resolved_preset.architecture, model_config)

    try:
        tokenizer = get_tokenizer(args.tokenizer_name, trust_remote_code=True)
        tokenizer_vocab_context = validate_tokenizer_vocab_size(
            tokenizer,
            args.tokenizer_name,
            model_config.vocab_size,
        )
    except ValueError as e:
        logger.error(f"Tokenizer validation failed: {e}")
        sys.exit(2)

    if args.mode == "autoep":
        try:
            validate_autoep_size(
                architecture=resolved_preset.architecture,
                autoep_size=autoep_size,
                num_experts=num_experts,
                world_size=world_size,
            )
            validate_autoep_structural_preset_available(
                ds_config["expert_parallel"]["preset_model"]
            )
        except ValueError as e:
            logger.error(f"AutoEP preflight failed: {e}")
            sys.exit(2)

    if rank == 0:
        logger.info(f"Mode: {args.mode}")
        logger.info(
            "Model preset: %s (%s, %s), %s layers%s, hidden=%s, %s experts",
            args.model,
            resolved_preset.display_name,
            resolved_preset.architecture,
            args.num_layers,
            " from --num_layers" if resolved_preset.num_layers_overridden else " original default",
            model_config.hidden_size,
            num_experts,
        )
        logger.info(
            "output_router_logits=%s (legacy include_router_aux_loss=%s)",
            output_router_logits,
            args.include_router_aux_loss,
        )
        logger.info(
            "Tokenizer %s: len=%s, vocab_size=%s, model_vocab_size=%s, exact_match=%s",
            args.tokenizer_name,
            tokenizer_vocab_context["tokenizer_len"],
            tokenizer_vocab_context["tokenizer_vocab_size"],
            tokenizer_vocab_context["model_vocab_size"],
            tokenizer_vocab_context["exact_vocab_match"],
        )
        if not args.init_weights_only:
            logger.info(
                f"HF dataset: {args.dataset_name!r}, "
                f"dataset_percentage={args.dataset_percentage}, "
                f"tokenizer={args.tokenizer_name!r}"
            )
        logger.info(f"Seq len: {args.seq_len}, Micro batch: {args.micro_batch_size}")
        logger.info(f"Grad accum: {args.grad_accum}, Steps: {args.steps}")

    # Build model with random weights
    if resolved_preset.architecture == "mixtral":
        model = AutoModelForCausalLM.from_config(model_config)
    elif resolved_preset.architecture == "qwen3_5_moe":
        model = Qwen3_5MoeForCausalLM(model_config)
    elif resolved_preset.architecture == "llama4":
        model = Llama4ForCausalLM(model_config)
    else:
        logger.error(f"Unsupported architecture: {resolved_preset.architecture}")
        sys.exit(2)

    init_weights_context = {
        "init_weights_path": None,
        "init_weights_sha256": None,
        "init_weights_loaded": False,
        "init_weights_schema_version": None,
    }

    # Optional: save pre-DeepSpeed init weights artifact
    if args.save_init_weights is not None:
        if args.init_weights_only or rank == 0:
            init_weights_context = save_init_weights_artifact(
                args.save_init_weights,
                model,
                args=args,
                model_config=model_config,
                rank=rank,
            )
            if rank == 0:
                logger.info(
                    f"Saved init weights artifact to {init_weights_context['init_weights_path']}"
                )
        if not args.init_weights_only and torch.distributed.is_initialized():
            torch.distributed.barrier()

    # Optional: load pre-DeepSpeed init weights artifact
    if args.load_init_weights is not None:
        if torch.distributed.is_initialized():
            # Rank-0 validates readability and broadcasts result to all ranks.
            read_ok = int(
                rank == 0
                and os.path.isfile(args.load_init_weights)
                and os.access(args.load_init_weights, os.R_OK)
            )
            check = torch.tensor([read_ok], device=torch.cuda.current_device())
            torch.distributed.broadcast(check, src=0)
            if check.item() != 1:
                logger.error(
                    f"Init weights path is not readable on rank 0: {args.load_init_weights}"
                )
                sys.exit(2)
            torch.distributed.barrier()

        try:
            init_weights_context = load_init_weights_artifact(
                args.load_init_weights,
                model,
                args=args,
                model_config=model_config,
            )
        except Exception as e:
            logger.error(f"Failed to load init weights artifact: {e}")
            sys.exit(2)
        if rank == 0:
            logger.info(
                "Loaded init weights artifact "
                f"{init_weights_context['init_weights_path']} "
                f"(sha256={init_weights_context['init_weights_sha256']})"
            )

    if args.init_weights_only:
        if rank == 0:
            logger.info("init_weights_only completed successfully.")
        return

    # Enable gradient checkpointing if requested (BEFORE deepspeed.initialize)
    if args.gradient_checkpointing == "on":
        model.gradient_checkpointing_enable()
        if rank == 0:
            logger.warning(
                "Gradient checkpointing enabled. tokens_per_expert will be inflated 2x "
                "by forward recomputation. Router logit hooks run 4x per layer."
            )

    # Initialize DeepSpeed engine
    try:
        if args.mode == "autoep":
            # AutoEP: do NOT pass optimizer; let DS build from config for MoE param groups
            engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                config=ds_config,
                model_parameters=model.parameters(),
            )
        else:
            # ZeRO-3 leaf: same pattern (DS builds optimizer from config)
            engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                config=ds_config,
                model_parameters=model.parameters(),
            )
    except Exception as e:
        logger.error(f"deepspeed.initialize() failed: {e}")
        sys.exit(2)

    if rank == 0:
        logger.info(f"DeepSpeed engine initialized. dp_world_size={engine.dp_world_size}")

    # Post-init validation
    gc_enabled = args.gradient_checkpointing == "on"
    if args.mode == "autoep":
        val_result = validate_autoep_engine(
            engine, autoep_size, num_experts, load_balance_coeff, gc_enabled
        )
    else:
        val_result = validate_zero3_leaf_engine(engine)

    if rank == 0:
        for w in val_result.get("warnings", []):
            logger.warning(f"Validation: {w}")
        for e in val_result.get("errors", []):
            logger.error(f"Validation: {e}")
        if not val_result["valid"]:
            logger.error("Post-init validation failed.")
            sys.exit(2)
        logger.info("Post-init validation passed.")
        if args.mode == "autoep":
            logger.info(
                f"Expert params: local={val_result['local_expert_param_numel']:,}, "
                f"global_est={val_result['global_expert_param_numel_est']:,}, "
                f"partition_ratio={val_result['expert_partition_ratio']:.1f}, "
                f"use_grouped_mm={val_result['use_grouped_mm']}"
            )

    # Checkpoint load (optional)
    start_step = 0
    if args.load_checkpoint is not None:
        load_path, client_state = engine.load_checkpoint(args.load_checkpoint)
        if load_path is None:
            logger.error(
                f"Failed to load checkpoint from {args.load_checkpoint}"
            )
            sys.exit(2)
        start_step = client_state.get("step", 0) if client_state else 0
        if rank == 0:
            logger.info(f"Loaded checkpoint from {load_path}, starting at step {start_step}")

    # Get DP rank for batch generator
    import deepspeed.comm as dist_comm

    dp_rank = dist_comm.get_rank(engine.data_parallel_group)
    dp_world_size = engine.dp_world_size

    # Hugging Face text batches (DistributedSampler over DP ranks)
    try:
        batch_gen = build_hf_batch_generator(
            dataset_name=args.dataset_name,
            dataset_percentage=args.dataset_percentage,
            tokenizer_name=args.tokenizer_name,
            expected_vocab_size=model_config.vocab_size,
            seq_len=args.seq_len,
            micro_batch_size=args.micro_batch_size,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            seed=args.seed,
            rank=rank,
            hf_num_dataloader_workers=args.hf_num_dataloader_workers,
        )
    except ValueError as e:
        logger.error(f"HF dataset setup failed: {e}")
        sys.exit(2)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Collect run metadata
    metadata = collect_run_metadata(
        args.mode,
        args,
        engine,
        val_result,
        resolved_autoep_size=autoep_size if args.mode == "autoep" else None,
        init_weights_context=init_weights_context,
    )

    # Setup metrics logger
    metrics_logger = MetricsLogger(args.metrics_out, rank)

    # Determine loss objective tag
    loss_tag = "ce_plus_aux" if output_router_logits else "ce_only"
    aux_loss_coef = float(getattr(model_config, "router_aux_loss_coef", 0.0))

    # Static metric fields from validation
    static_metrics = {}
    if args.mode == "autoep":
        static_metrics.update(
            {
                "local_expert_param_numel": val_result["local_expert_param_numel"],
                "global_expert_param_numel_est": val_result[
                    "global_expert_param_numel_est"
                ],
                "expert_partition_ratio": val_result["expert_partition_ratio"],
                "use_grouped_mm": val_result["use_grouped_mm"],
            }
        )
    else:
        static_metrics.update(
            {
                "local_expert_param_numel": "",
                "global_expert_param_numel_est": "",
                "expert_partition_ratio": "",
                "use_grouped_mm": "",
            }
        )

    # Training loop
    if rank == 0:
        logger.info(f"Starting training for {args.steps} optimizer steps (warmup={args.warmup_steps})...")

    tokens_per_microstep = args.seq_len * args.micro_batch_size
    aux_loss_missing_warned = False

    for step in range(start_step, args.steps):
        torch.cuda.synchronize()
        step_start = time.time()

        last_loss = None
        last_ce_loss = None
        last_aux_loss = None

        for accum_idx in range(args.grad_accum):
            batch = batch_gen.get_batch(step, accum_idx)
            batch_dict = {
                "input_ids": batch.input_ids.to(engine.device),
                "attention_mask": batch.attention_mask.to(engine.device),
                "labels": batch.labels.to(engine.device),
            }

            outputs = engine(**batch_dict)
            loss = outputs.loss
            aux_loss = getattr(outputs, "aux_loss", None)
            if aux_loss is not None:
                ce_loss = loss - aux_loss.to(loss.device) * aux_loss_coef
            else:
                ce_loss = loss
                if output_router_logits and not aux_loss_missing_warned and rank == 0:
                    logger.warning(
                        "output_router_logits is enabled but model outputs did not "
                        "include aux_loss; loss_aux metrics will be blank."
                    )
                    aux_loss_missing_warned = True
            last_loss = loss.detach().clone()
            last_ce_loss = ce_loss.detach().clone()
            last_aux_loss = (
                aux_loss.detach().clone() if torch.is_tensor(aux_loss) else None
            )

            engine.backward(loss)
            engine.step()

        torch.cuda.synchronize()
        step_end = time.time()
        iter_time = step_end - step_start

        # Reduce loss (DP-mean) - use last microstep loss
        reduced_total_loss = reduce_loss(
            last_loss, dp_world_size, group=engine.data_parallel_group
        )
        reduced_ce_loss = reduce_loss(
            last_ce_loss, dp_world_size, group=engine.data_parallel_group
        )
        reduced_aux_loss = (
            reduce_loss(last_aux_loss, dp_world_size, group=engine.data_parallel_group)
            if last_aux_loss is not None
            else None
        )

        # Non-finite loss check (every step including warmup)
        if not math.isfinite(reduced_total_loss) or not math.isfinite(reduced_ce_loss):
            if rank == 0:
                logger.error(
                    "Non-finite loss at step %s: total=%s ce=%s",
                    step,
                    reduced_total_loss,
                    reduced_ce_loss,
                )
            sys.exit(3)
        if reduced_aux_loss is not None and not math.isfinite(reduced_aux_loss):
            if rank == 0:
                logger.error(
                    "Non-finite aux loss at step %s: aux=%s",
                    step,
                    reduced_aux_loss,
                )
            sys.exit(3)

        # Reset peak memory stats after warmup
        if step == args.warmup_steps - 1:
            torch.cuda.reset_peak_memory_stats()

        # Log metrics for steps >= warmup_steps
        if step >= args.warmup_steps and step % args.log_interval == 0:
            # Reduce timing (max across ranks)
            max_iter_time = reduce_max(iter_time)

            # Memory stats
            mem_allocated = torch.cuda.memory_allocated()
            mem_peak_allocated = torch.cuda.max_memory_allocated()
            mem_peak_reserved = torch.cuda.max_memory_reserved()

            # Throughput
            total_tokens_this_step = tokens_per_microstep * args.grad_accum
            tokens_per_sec = total_tokens_this_step / max_iter_time if max_iter_time > 0 else 0
            global_tokens_per_sec = (
                args.seq_len
                * args.micro_batch_size
                * args.grad_accum
                * dp_world_size
                / max_iter_time
                if max_iter_time > 0
                else 0
            )

            step_metrics = {
                "step": step,
                "loss_ce": reduced_ce_loss,
                "loss_total": reduced_total_loss,
                "loss_aux": "" if reduced_aux_loss is None else reduced_aux_loss,
                "loss_objective_tag": loss_tag,
                "iter_time_sec": max_iter_time,
                "tokens_per_sec": tokens_per_sec,
                "global_tokens_per_sec": global_tokens_per_sec,
                "cuda_memory_allocated_bytes": mem_allocated,
                "cuda_peak_memory_allocated_bytes": mem_peak_allocated,
                "cuda_peak_memory_reserved_bytes": mem_peak_reserved,
                **static_metrics,
            }
            metrics_logger.log_step(step_metrics)

            if rank == 0:
                aux_msg = (
                    f", aux={reduced_aux_loss:.6f}"
                    if reduced_aux_loss is not None
                    else ""
                )
                logger.info(
                    f"Step {step}: loss_total={reduced_total_loss:.6f}, "
                    f"loss_ce={reduced_ce_loss:.6f}{aux_msg}, "
                    f"time={max_iter_time:.3f}s, "
                    f"global_tps={global_tokens_per_sec:.0f}"
                )

    metrics_logger.close()

    # Checkpoint save (optional)
    if args.save_checkpoint is not None:
        engine.save_checkpoint(
            args.save_checkpoint, client_state={"step": args.steps}
        )
        if rank == 0:
            logger.info(f"Saved checkpoint to {args.save_checkpoint}")

    # Write run metadata (rank 0 only)
    if rank == 0:
        write_run_metadata(metadata, args.run_metadata_out)
        logger.info(f"Metrics written to {args.metrics_out}")
        logger.info(f"Metadata written to {args.run_metadata_out}")
        logger.info("Training complete.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        logging.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
