# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Export Megatron-Bridge's own SFT samples, for rdsp to train on the same data.

Runs inside the NeMo container. Builds the training dataset exactly as a
Megatron-Bridge recipe does (same Hugging Face source and schema adapter,
same processor and image pixel limits, same collate, sequence length and
padding), then collates each sample alone in index order. With
dataloader_type="single" Megatron reads samples in that same order for any
data-parallel width, and a global microbatch is consecutive samples, so rdsp
reading these files in order sees the same rows in the same microbatches.

Per sample it keeps what the Hugging Face model needs (input ids, attention
mask, mm_token_type_ids, pixels, image grid) and the labels in the Hugging
Face convention: unshifted, -100 where the loss mask is 0. Pixels are stored
in bf16, the dtype the model casts them to. One file per training step.

    python export_bridge_batches.py --out /workspace/cord_steps --steps 50 \\
        --recipe qwen35_vl_9b_sft_4gpu_h100_bf16_config \\
        model.seq_length=2048 dataset.seq_length=2048 train.global_batch_size=64 \\
        dataset.source.dataset_name=cord_v2
"""

import argparse
import json
import os
import sys

import torch

IGNORE = -100


def to_rdsp_sample(batch: dict) -> dict:
    """One collated row (batch of 1) from the Bridge Qwen-VL collate: its
    labels are shifted left (label t is the target of position t), so the
    target of position t moves to t + 1 for a loss that shifts itself."""
    shifted = batch["labels"].masked_fill(batch["loss_mask"] == 0, IGNORE)
    labels = torch.full_like(shifted, IGNORE)
    labels[:, 1:] = shifted[:, :-1]
    visual = batch["visual_inputs"]
    return {
        "input_ids": batch["input_ids"].clone(),
        "attention_mask": batch["attention_mask"].clone(),
        "mm_token_type_ids": batch["mm_token_type_ids"].clone(),
        "pixel_values": visual.pixel_values.to(torch.bfloat16),
        "image_grid_thw": visual.image_grid_thw.clone(),
        "labels": labels,
        "tokens": int(batch["loss_mask"].sum()),
    }


def save_step(out: str, step: int, samples: list) -> None:
    os.makedirs(out, exist_ok=True)
    torch.save(samples, os.path.join(out, f"step_{step:05d}.pt"))


def _bridge_config(recipe_name: str, overrides: list):
    """The recipe's config with the training run's overrides applied."""
    bridge = os.environ.get("BRIDGE_ROOT", "/opt/Megatron-Bridge")
    sys.path.insert(0, os.path.join(bridge, "scripts", "training"))
    from recipe_runner import apply_cli_overrides, load_recipe

    cfg = apply_cli_overrides(load_recipe(recipe_name), overrides)
    if getattr(cfg.dataset, "dataloader_type", "single") != "single":
        raise SystemExit(
            "set dataset.dataloader_type=single: other samplers reorder per data-parallel rank"
        )
    return cfg


def _bridge_dataset(cfg, n_samples: int):
    """The training dataset, built the way training builds it."""
    from megatron.bridge.data.base import DatasetBuildContext
    from megatron.bridge.data.builders.direct_hf_sft import DirectHFSFTDatasetBuilder

    context = DatasetBuildContext(train_samples=n_samples, valid_samples=0, test_samples=0)
    train, _, _ = DirectHFSFTDatasetBuilder(cfg.dataset).build(context)
    return train


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--recipe", required=True)
    p.add_argument("--steps", type=int, default=50)
    args, overrides = p.parse_known_args(argv)

    cfg = _bridge_config(args.recipe, overrides)
    gbs = int(cfg.train.global_batch_size)
    dataset = _bridge_dataset(cfg, args.steps * gbs)
    stats = {
        "steps": args.steps,
        "global_batch": gbs,
        "seq_length": int(cfg.dataset.seq_length),
        "supervised_tokens": 0,
        "real_tokens": 0,
        "vision_tokens": 0,
        "full_rows": 0,
    }
    for step in range(args.steps):
        samples = []
        for i in range(step * gbs, (step + 1) * gbs):
            example = dataset[i]
            batch = dataset.collate_fn([example])
            sample = to_rdsp_sample(batch)
            samples.append(sample)
            stats["supervised_tokens"] += sample["tokens"]
            stats["real_tokens"] += int(sample["attention_mask"].sum())
            stats["vision_tokens"] += int(sample["image_grid_thw"].prod(-1).sum()) // 4
            # a row filling the whole sequence may have been cut short
            stats["full_rows"] += int(sample["attention_mask"][0, -1] == 1)
        save_step(args.out, step, samples)
        print(f"step {step}: {gbs} samples", flush=True)
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
