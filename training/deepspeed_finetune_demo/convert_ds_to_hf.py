#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert DeepSpeed ZeRO-2 + AutoEP model weights back to HuggingFace format.

With AutoEP (expert parallelism), each rank holds only its local expert shard.
The training script saves:
  - Rank 0: full state dict (non-expert params + local experts 0..E_local-1)
  - Rank 1..N-1: only expert shard params (w1, w2, w3 as 3D tensors)

This script:
  1. Loads model_weights.pt from all EP ranks
  2. Takes non-expert params from rank 0
  3. Concatenates expert shards (w1, w2, w3) across ranks
  4. Unpacks grouped 3D tensors to per-expert module_list format
  5. Remaps AutoEP router keys back to HF gate keys
  6. Saves as a standard HF model

Usage:
    python convert_ds_to_hf.py \
        --ds_checkpoint output_moonlight_muon \
        --original_model moonshotai/Moonlight-16B-A3B \
        --output_dir hf_model_muon \
        --ep_size 8
"""

import argparse
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_rank_state_dict(rank_dir):
    """Load model weights from a single rank's directory."""
    weights_file = os.path.join(rank_dir, "model_weights.pt")
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"Model weights not found: {weights_file}")
    return torch.load(weights_file, map_location="cpu", weights_only=False)


# Regex to detect AutoEP expert grouped tensor keys
# Pattern: model.layers.{L}.mlp.experts.w{1,2,3}
_EXPERT_W_RE = re.compile(r"^(model\.layers\.\d+\.mlp)\.experts\.(w[123])$")

# Pattern for AutoEP router keys. AutoEP wraps the gate in a router module:
# model.layers.{L}.mlp.router.gate.weight -> model.layers.{L}.mlp.gate.weight
# model.layers.{L}.mlp.router.e_score_correction_bias -> model.layers.{L}.mlp.gate.e_score_correction_bias
_ROUTER_RE = re.compile(r"^(model\.layers\.\d+\.mlp)\.router\.(.+)$")

# Mapping: AutoEP w names -> HF projection names
_W_TO_PROJ = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


def merge_and_unpack(rank0_sd, expert_shard_sds, ep_size):
    """Merge rank 0's full state dict with expert shards from other ranks.

    Args:
        rank0_sd: Full state dict from rank 0 (non-expert + expert shard 0)
        expert_shard_sds: List of expert-only state dicts from ranks 1..N-1
        ep_size: Number of EP ranks
    """
    hf_state_dict = {}

    for key, value in rank0_sd.items():
        m_expert = _EXPERT_W_RE.match(key)
        m_router = _ROUTER_RE.match(key)

        if m_expert:
            prefix = m_expert.group(1)  # e.g., model.layers.2.mlp
            w_name = m_expert.group(2)  # e.g., w1

            # Concatenate: rank 0's shard + shards from ranks 1..N-1
            shards = [value]
            for rank_sd in expert_shard_sds:
                shards.append(rank_sd[key])
            full_tensor = torch.cat(shards, dim=0)  # [E_total, ...]

            # Unpack to per-expert module_list format
            proj_name = _W_TO_PROJ[w_name]
            num_experts = full_tensor.shape[0]
            for e in range(num_experts):
                hf_key = f"{prefix}.experts.{e}.{proj_name}.weight"
                hf_state_dict[hf_key] = full_tensor[e]
        elif m_router:
            prefix = m_router.group(1)
            rest = m_router.group(2)
            # Remap AutoEP router keys back to HF gate module keys.
            # AutoEP router has: gate (sub-module) and e_score_correction_bias (direct attr).
            # HF model has: gate.weight and gate.e_score_correction_bias (both under gate).
            # So: mlp.router.gate.weight -> mlp.gate.weight (strip "router.")
            #     mlp.router.e_score_correction_bias -> mlp.gate.e_score_correction_bias
            if rest.startswith("gate."):
                hf_key = f"{prefix}.{rest}"
            else:
                hf_key = f"{prefix}.gate.{rest}"
            hf_state_dict[hf_key] = value
        else:
            # Non-expert, non-router param: take from rank 0 as-is
            hf_state_dict[key] = value

    return hf_state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_checkpoint",
        type=str,
        required=True,
        help="Path to DeepSpeed checkpoint output directory",
    )
    parser.add_argument(
        "--original_model", type=str, required=True, help="Original HF model name/path"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for HF model"
    )
    parser.add_argument(
        "--ep_size",
        type=int,
        default=8,
        help="Expert parallelism size (number of ranks)",
    )
    args = parser.parse_args()

    # Load rank 0 (full state dict)
    rank0_dir = os.path.join(args.ds_checkpoint, "0")
    if not os.path.isdir(rank0_dir):
        raise FileNotFoundError(f"Rank 0 directory not found: {rank0_dir}")
    print("Loading rank 0 (full state dict)...")
    rank0_sd = load_rank_state_dict(rank0_dir)
    print(f"  Rank 0: {len(rank0_sd)} keys")

    # Load expert shards from ranks 1..N-1
    expert_shard_sds = []
    for rank in range(1, args.ep_size):
        rank_dir = os.path.join(args.ds_checkpoint, str(rank))
        if not os.path.isdir(rank_dir):
            raise FileNotFoundError(f"Rank {rank} directory not found: {rank_dir}")
        print(f"Loading rank {rank} (expert shard only)...")
        sd = load_rank_state_dict(rank_dir)
        expert_shard_sds.append(sd)
        print(f"  Rank {rank}: {len(sd)} keys")

    # Merge and unpack
    print("Merging expert shards and unpacking to HF format...")
    hf_state_dict = merge_and_unpack(rank0_sd, expert_shard_sds, args.ep_size)
    print(f"Merged state dict: {len(hf_state_dict)} keys")

    # Free loaded state dicts to save memory
    del rank0_sd, expert_shard_sds

    # Load original model structure
    print(f"Loading original model: {args.original_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.original_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load the merged state dict
    missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)
    if missing:
        print(f"WARNING: {len(missing)} missing keys:")
        for k in missing[:20]:
            print(f"  Missing: {k}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys:")
        for k in unexpected[:20]:
            print(f"  Unexpected: {k}")
        if len(unexpected) > 20:
            print(f"  ... and {len(unexpected) - 20} more")
    if not missing and not unexpected:
        print("All keys matched perfectly!")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving HF model to: {args.output_dir}")
    model.save_pretrained(args.output_dir)

    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.original_model, trust_remote_code=True
    )
    tokenizer.save_pretrained(args.output_dir)

    # Copy custom modeling code (save_pretrained only copies config, not modeling code)
    import shutil

    code_dir = os.path.dirname(model.config.__class__.__module__.replace(".", "/"))
    # Find the actual source directory from the module's file path
    import importlib

    config_module = importlib.import_module(model.config.__class__.__module__)
    src_dir = os.path.dirname(config_module.__file__)
    for fname in os.listdir(src_dir):
        if fname.endswith(".py") and not fname.startswith("__"):
            src_file = os.path.join(src_dir, fname)
            dst_file = os.path.join(args.output_dir, fname)
            if not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)
                print(f"  Copied custom code: {fname}")

    print("Done!")


if __name__ == "__main__":
    main()
