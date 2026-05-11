"""Create a shared initialization artifact for AutoEP comparison runs.

Run this as a normal Python script, not through the DeepSpeed launcher:

    python prepare_init_weights.py --model qwen3_5 --num_layers 8 \
        --output runs/qwen35/init/qwen35_l8_seed42.safetensors
"""

from __future__ import annotations

import argparse
import logging
import random

import numpy as np
import torch

from init_weights import save_init_weights_artifact
from train import MODEL_PRESETS, build_model, build_model_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create shared init weights artifact")
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), default="qwen3_5")
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        required=True,
        help="Output .safetensors path. A sidecar *_meta.json file is also written.",
    )
    args = parser.parse_args()
    if not args.output.endswith(".safetensors"):
        parser.error("--output path must end with '.safetensors'.")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    seed_everything(args.seed)

    architecture = MODEL_PRESETS[args.model]["architecture"]
    model_config = build_model_config(architecture, args.num_layers)
    if args.num_layers is None:
        args.num_layers = int(model_config.num_hidden_layers)

    model = build_model(architecture, model_config)
    context = save_init_weights_artifact(
        args.output,
        model,
        args=args,
        model_config=model_config,
        rank=0,
    )
    logger.info("Saved init weights to %s", context["init_weights_path"])
    logger.info("sha256=%s", context["init_weights_sha256"])


if __name__ == "__main__":
    main()
