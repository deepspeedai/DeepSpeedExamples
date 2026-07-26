# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Phase II of RAC: one-shot pruning against a chosen calibration set.

Everything except ``--calibration`` is a standard SparseGPT/Wanda run; passing
``--calibration rac --traces <file>`` is what turns it into Reasoning-Aware
Compression.

Example -- 50% unstructured sparsity with RAC calibration::

    python prune.py \\
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \\
        --calibration rac --traces traces/math_1.5b.jsonl \\
        --sparsity 0.5 --nsamples 512 --seqlen 2048 \\
        --output checkpoints/r1-qwen-1.5b-rac-50

The model is held on CPU and streamed to the accelerator one transformer block
at a time, so peak device memory is set by the largest block rather than by the
model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rac import (
    build_calibration_samples,
    dtype_kwargs,
    layerwise_sparsity,
    magnitude_prune,
    model_sparsity,
    select_blocks_by_thirds,
    sequential_prune,
)
from rac.calibration import CALIBRATION_SOURCES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    model = parser.add_argument_group("model")
    model.add_argument("--model", required=True, help="Dense checkpoint to prune.")
    model.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    model.add_argument("--trust-remote-code", action="store_true")
    model.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    calib = parser.add_argument_group("calibration")
    calib.add_argument("--calibration", default="rac", choices=list(CALIBRATION_SOURCES),
                       help="rac: prompts + on-policy CoT traces (the method). "
                            "prompt: prompts only. c4: generic web text.")
    calib.add_argument("--traces", default=None,
                       help="JSONL from collect_traces.py. Required for --calibration rac; "
                            "with --calibration prompt it reuses the same prompts without CoT.")
    calib.add_argument("--dataset", default=None,
                       help="HF dataset (or .jsonl) for --calibration prompt / c4.")
    calib.add_argument("--dataset-config", default=None)
    calib.add_argument("--dataset-split", default="train")
    calib.add_argument("--prompt-column", default="problem")
    calib.add_argument("--system-prompt", default=None)
    calib.add_argument("--no-chat-template", action="store_true")
    calib.add_argument("--nsamples", type=int, default=512,
                       help="Calibration windows. nsamples * seqlen = calibration tokens; "
                            "512 x 2048 = 1M, as in the paper.")
    calib.add_argument("--seqlen", type=int, default=2048, help="Tokens per calibration window.")
    calib.add_argument("--seed", type=int, default=0)

    prune = parser.add_argument_group("pruning")
    prune.add_argument("--pruning-method", default="sparsegpt",
                       choices=["sparsegpt", "wanda", "magnitude"])
    prune.add_argument("--sparsity", type=float, default=0.5,
                       help="Per-layer unstructured sparsity; ignored with --prune-n/--prune-m.")
    prune.add_argument("--prune-n", type=int, default=0,
                       help="Semi-structured n:m sparsity, e.g. --prune-n 2 --prune-m 4.")
    prune.add_argument("--prune-m", type=int, default=0)
    prune.add_argument("--scope", default="all", choices=["all", "mlp"],
                       help="Which projections to prune within a block.")
    prune.add_argument("--layer-thirds", default="1,2,3",
                       help="Which thirds of the decoder stack to prune, e.g. '1,3' for the "
                            "first and last third (Table 6 of the paper).")
    prune.add_argument("--batch-size", type=int, default=1,
                       help="Calibration samples per forward pass.")
    prune.add_argument("--blocksize", type=int, default=128, help="SparseGPT column block size.")
    prune.add_argument("--percdamp", type=float, default=0.01,
                       help="Hessian damping, as a fraction of its mean diagonal.")

    parser.add_argument("--output", required=True, help="Directory for the pruned checkpoint.")
    parser.add_argument("--save-report", action="store_true",
                        help="Also write per-layer sparsity to the output directory.")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    if bool(args.prune_n) != bool(args.prune_m):
        raise ValueError("--prune-n and --prune-m must be given together (e.g. 2 and 4).")
    if args.prune_n and args.prune_n >= args.prune_m:
        raise ValueError(f"--prune-n ({args.prune_n}) must be smaller than "
                         f"--prune-m ({args.prune_m}).")
    if not args.prune_n and not 0.0 < args.sparsity < 1.0:
        raise ValueError(f"--sparsity must be in (0, 1), got {args.sparsity}.")
    if args.calibration == "rac" and not args.traces:
        raise ValueError("--calibration rac requires --traces; run collect_traces.py first.")


def main() -> None:
    args = parse_args()
    validate(args)

    print(f"[rac] loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
        **dtype_kwargs(getattr(torch, args.dtype)),
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    thirds = [int(t) for t in args.layer_thirds.replace(",", " ").split()]
    num_blocks = model.config.num_hidden_layers
    block_indices = select_blocks_by_thirds(num_blocks, thirds)
    target = (f"{args.prune_n}:{args.prune_m}" if args.prune_n else f"{args.sparsity:.0%}")
    print(f"[rac] pruning {len(block_indices)}/{num_blocks} blocks to {target} "
          f"with {args.pruning_method}, scope={args.scope}")

    started = time.time()
    stats = []

    if args.pruning_method == "magnitude":
        print("[rac] magnitude pruning ignores calibration data")
        magnitude_prune(
            model,
            sparsity=args.sparsity,
            prune_n=args.prune_n,
            prune_m=args.prune_m,
            scope=args.scope,
            block_indices=block_indices,
        )
    else:
        print(f"[rac] building '{args.calibration}' calibration set: "
              f"{args.nsamples} x {args.seqlen} = {args.nsamples * args.seqlen:,} tokens")
        samples = build_calibration_samples(
            args.calibration,
            tokenizer,
            nsamples=args.nsamples,
            seqlen=args.seqlen,
            seed=args.seed,
            traces=args.traces,
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            prompt_column=args.prompt_column,
            system_prompt=args.system_prompt,
            use_chat_template=not args.no_chat_template,
        )
        stats = sequential_prune(
            model,
            samples,
            method=args.pruning_method,
            sparsity=args.sparsity,
            prune_n=args.prune_n,
            prune_m=args.prune_m,
            scope=args.scope,
            block_indices=block_indices,
            device=args.device,
            blocksize=args.blocksize,
            percdamp=args.percdamp,
            batch_size=args.batch_size,
        )

    elapsed = time.time() - started
    realised = model_sparsity(model, scope=args.scope, block_indices=block_indices)
    overall = model_sparsity(model, scope="all")
    print(f"[rac] done in {elapsed / 60:.1f} min; realised sparsity {realised:.2%} over the "
          f"pruned layers, {overall:.2%} over all decoder-block weights")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)

    metadata = {
        "base_model": args.model,
        "pruning_method": args.pruning_method,
        "calibration": args.calibration,
        "traces": args.traces,
        "calibration_tokens": args.nsamples * args.seqlen,
        "sparsity": args.sparsity,
        "prune_n": args.prune_n,
        "prune_m": args.prune_m,
        "scope": args.scope,
        "layer_thirds": thirds,
        "pruned_blocks": block_indices,
        "realised_sparsity_pruned_layers": realised,
        "realised_sparsity_all_blocks": overall,
        "wall_clock_seconds": elapsed,
        "reconstruction_loss": sum(s["loss"] for s in stats) if stats else None,
    }
    (output / "rac_pruning.json").write_text(json.dumps(metadata, indent=2) + "\n")

    if args.save_report:
        report = dict(layerwise_sparsity(model))
        (output / "rac_layer_sparsity.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"[rac] pruned checkpoint written to {output}")


if __name__ == "__main__":
    main()
