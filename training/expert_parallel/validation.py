"""Post-init validation and runtime metadata collection for AutoEP example."""

import importlib.metadata
import os
import socket
import subprocess
from typing import Any


def _native_sparse_moe_block_types() -> tuple[type, ...]:
    """HF MoE block classes that DeepSpeed AutoEP replaces (subset used in this example)."""
    types: list[type] = []
    try:
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

        types.append(MixtralSparseMoeBlock)
    except ImportError:
        pass
    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeSparseMoeBlock,
        )

        types.append(Qwen3_5MoeSparseMoeBlock)
    except ImportError:
        pass
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

        types.append(Qwen3MoeSparseMoeBlock)
    except ImportError:
        pass
    return tuple(types)


def _is_native_sparse_moe_block(module: Any) -> bool:
    return isinstance(module, _native_sparse_moe_block_types())


def validate_autoep_engine(
    engine: Any,
    autoep_size: int,
    num_experts: int,
    load_balance_coeff: float | None,
    gradient_checkpointing: bool,
) -> dict[str, Any]:
    """Post-init validation for AutoEP mode.

    Checks replacement integrity, param attributes, partition ratio, optimizer groups,
    and grouped GEMM backend status.
    """
    try:
        from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
    except ImportError:
        return {
            "valid": False,
            "local_expert_param_numel": 0,
            "global_expert_param_numel_est": 0,
            "expert_partition_ratio": 0.0,
            "use_grouped_mm": False,
            "warnings": [],
            "errors": [
                "Cannot import AutoEPMoELayer. "
                "Ensure DeepSpeed is installed from an AutoEP-enabled branch."
            ],
        }

    errors = []
    warnings = []

    # Check engine.has_moe_layers
    if not getattr(engine, "has_moe_layers", False):
        errors.append("engine.has_moe_layers is False; expected True for AutoEP mode.")

    # Replacement integrity: AutoEPMoELayer present, native HF MoE blocks absent
    autoep_layers = []
    original_moe_blocks = []

    for name, module in engine.module.named_modules():
        if isinstance(module, AutoEPMoELayer):
            autoep_layers.append(name)
        if _is_native_sparse_moe_block(module):
            original_moe_blocks.append(name)

    if not autoep_layers:
        errors.append("No AutoEPMoELayer modules found in the model.")
    if original_moe_blocks:
        errors.append(
            f"Found unreplaced native MoE block modules: {original_moe_blocks[:3]}..."
        )

    # Expert param attributes
    local_expert_numel = 0
    for name, param in engine.module.named_parameters():
        if hasattr(param, "allreduce"):
            if not param.allreduce:
                # Expert parameter
                local_expert_numel += param.numel()
                if not hasattr(param, "group_name") or not param.group_name:
                    errors.append(
                        f"Expert param {name} has allreduce=False but no group_name set."
                    )

    # Check router params have allreduce=True
    for name, param in engine.module.named_parameters():
        if "gate" in name or "router" in name:
            if hasattr(param, "allreduce") and not param.allreduce:
                errors.append(f"Router param {name} has allreduce=False; expected True.")

    # Partition ratio
    global_expert_numel_est = local_expert_numel * autoep_size
    partition_ratio = (
        global_expert_numel_est / local_expert_numel
        if local_expert_numel > 0
        else 0.0
    )

    # Verify num_local_experts on AutoEP layers
    for name, module in engine.module.named_modules():
        if isinstance(module, AutoEPMoELayer):
            expected_local = num_experts // autoep_size
            actual_local = getattr(module, "num_local_experts", None)
            if actual_local is not None and actual_local != expected_local:
                errors.append(
                    f"{name}: num_local_experts={actual_local}, "
                    f"expected {expected_local} (num_experts={num_experts}/autoep_size={autoep_size})"
                )
            break

    # Check optimizer param groups for MoE split
    has_moe_group = False
    if hasattr(engine, "optimizer") and hasattr(engine.optimizer, "param_groups"):
        for pg in engine.optimizer.param_groups:
            if pg.get("moe", False):
                has_moe_group = True
                break
    if not has_moe_group:
        warnings.append(
            "No optimizer param group with moe=True found. "
            "Expert parameters may not be using expert-data-parallel reduction."
        )

    # Grouped GEMM backend
    use_grouped_mm = False
    for _name, module in engine.module.named_modules():
        if isinstance(module, AutoEPMoELayer):
            experts = getattr(module, "experts", None)
            if experts is not None:
                use_grouped_mm = getattr(experts, "use_grouped_mm", False)
            break

    if not use_grouped_mm:
        warnings.append(
            "Sequential expert for-loop fallback is active (torch._grouped_mm unavailable). "
            "This is functional but substantially slower than grouped GEMM."
        )

    # Load balance coefficient warning
    if load_balance_coeff is not None:
        warnings.append(
            f"load_balance_coeff={load_balance_coeff} is set but the expert_bias "
            "update pre-hook is not yet implemented. expert_bias will remain at zero."
        )

    # Gradient checkpointing warning
    if gradient_checkpointing:
        warnings.append(
            "Gradient checkpointing is enabled. tokens_per_expert counts will be "
            "inflated 2x by forward recomputation. Router logit hooks run 4x per layer."
        )

    return {
        "valid": len(errors) == 0,
        "local_expert_param_numel": local_expert_numel,
        "global_expert_param_numel_est": global_expert_numel_est,
        "expert_partition_ratio": partition_ratio,
        "use_grouped_mm": use_grouped_mm,
        "warnings": warnings,
        "errors": errors,
    }


def validate_zero3_leaf_engine(engine: Any) -> dict[str, Any]:
    """Post-init validation for ZeRO-3 leaf baseline mode."""
    errors = []
    warnings = []

    # Native HF MoE blocks should be present (not replaced by AutoEP)
    native_types = _native_sparse_moe_block_types()
    if not native_types:
        return {
            "valid": False,
            "warnings": [],
            "errors": ["Cannot import any known HF SparseMoeBlock class from transformers."],
        }

    moe_blocks = []
    for name, module in engine.module.named_modules():
        if _is_native_sparse_moe_block(module):
            moe_blocks.append(name)

    if not moe_blocks:
        errors.append("No native HF SparseMoeBlock modules found in the model.")

    # ZeRO-3 leaf does NOT use DeepSpeed MoE; has_moe_layers should be False
    if getattr(engine, "has_moe_layers", False):
        warnings.append(
            "engine.has_moe_layers is True in ZeRO-3 leaf mode. "
            "This is unexpected but not necessarily an error."
        )

    # Check ZeRO-3 partitioning is active
    if hasattr(engine, "zero_optimization_partition_weights"):
        if not engine.zero_optimization_partition_weights():
            errors.append("ZeRO-3 weight partitioning is not active.")

    return {
        "valid": len(errors) == 0,
        "warnings": warnings,
        "errors": errors,
    }


def _run_git_command(cwd: str, *args: str) -> str:
    """Run a git command and return stripped stdout, or 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git"] + list(args),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def collect_run_metadata(
    mode: str,
    args: Any,
    engine: Any,
    validation_result: dict[str, Any],
    resolved_autoep_size: int | None = None,
    init_weights_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect runtime metadata for reproducibility."""
    import torch
    import transformers
    import deepspeed

    # Git SHAs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    examples_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    ds_root = os.path.join(examples_root, "..", "DeepSpeed")
    if not os.path.isdir(ds_root):
        # Try common dev workspace layout
        ds_root = os.path.join(os.path.dirname(examples_root), "DeepSpeed")

    ds_sha = _run_git_command(ds_root, "rev-parse", "HEAD")
    ds_branch = _run_git_command(ds_root, "branch", "--show-current")
    examples_sha = _run_git_command(examples_root, "rev-parse", "HEAD")

    # Package versions
    try:
        ds_version = importlib.metadata.version("deepspeed")
    except Exception:
        ds_version = "unknown"

    # NCCL version
    try:
        nccl_ver = torch.cuda.nccl.version()
        nccl_version = ".".join(str(v) for v in nccl_ver) if isinstance(nccl_ver, tuple) else str(nccl_ver)
    except Exception:
        nccl_version = "unknown"

    # Driver version
    try:
        driver_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        driver_version = driver_result.stdout.strip().split("\n")[0] if driver_result.returncode == 0 else "unknown"
    except Exception:
        driver_version = "unknown"

    # Effective tokens per update
    dp_ws = engine.dp_world_size if hasattr(engine, "dp_world_size") else 1
    seq_len = getattr(args, "seq_len", 128)
    mbs = getattr(args, "micro_batch_size", 2)
    ga = getattr(args, "grad_accum", 1)
    effective_tokens = seq_len * mbs * ga * dp_ws

    init_ctx = init_weights_context or {}
    init_weights_path = init_ctx.get("init_weights_path")
    init_weights_sha256 = init_ctx.get("init_weights_sha256")
    init_weights_loaded = bool(init_ctx.get("init_weights_loaded", False))
    init_weights_schema_version = init_ctx.get("init_weights_schema_version")

    return {
        "mode": mode,
        "deepspeed_git_sha": ds_sha,
        "deepspeed_branch": ds_branch,
        "deepspeedexamples_git_sha": examples_sha,
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "deepspeed_version": ds_version,
        "deepspeed_file": getattr(deepspeed, "__file__", "unknown"),
        "transformers_file": getattr(transformers, "__file__", "unknown"),
        "cuda_version": torch.version.cuda or "unknown",
        "nccl_version": nccl_version,
        "driver_version": driver_version,
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "dp_world_size": dp_ws,
        "autoep_size": resolved_autoep_size,
        "num_gpus": torch.cuda.device_count(),
        "hostname": socket.gethostname(),
        "effective_tokens_per_update": effective_tokens,
        "validation": validation_result,
        "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "allow_untested_versions": getattr(args, "allow_untested_versions", False),
        "init_weights_path": init_weights_path,
        "init_weights_sha256": init_weights_sha256,
        "init_weights_loaded": init_weights_loaded,
        "init_weights_schema_version": init_weights_schema_version,
    }
