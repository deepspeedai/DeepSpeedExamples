#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""grids.py <out.json> [count]: [t, h, w] patch grids of real CORD-v2 receipts after the Qwen3.5
image processor, with the pixel cap whose mean is closest to our training rows: 800.7 merged
tokens (3,203 patches) per image in Megatron-Bridge's export, 2,562,208 vision tokens over 3,200
rows (bridge_vs_rdsp/9b_tp4pp2dp1/logs/export.log). The processor's default cap gives about
7,800 patches per image, which those 2,048-token rows (none full) cannot have held."""

import json
import sys

import datasets
from transformers import AutoProcessor

TARGET = 2562208 / 3200 * 4  # patches per image in training
out, count = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 64
proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-27B")
data = datasets.load_dataset("naver-clova-ix/cord-v2", split="train").shuffle(seed=0)
images = [row["image"].convert("RGB") for row in data.select(range(count))]
best = None
for cap in (2_000_000, 1_500_000, 1_200_000, 1_000_000, 900_000, 800_000, 700_000, 600_000):
    grids = []
    for im in images:
        thw = proc.image_processor(
            images=im, return_tensors="pt", size={"longest_edge": cap, "shortest_edge": 56 * 56}
        )["image_grid_thw"]
        grids.append([int(x) for x in thw[0]])
    mean = sum(t * h * w for t, h, w in grids) / len(grids)
    print(f"cap {cap:,} pixels: patches per image mean {mean:.0f}")
    if best is None or abs(mean - TARGET) < abs(best[1] - TARGET):
        best = (cap, mean, grids)
cap, mean, grids = best
json.dump(grids, open(out, "w"))
patches = [t * h * w for t, h, w in grids]
print(
    f"chosen cap {cap:,}: {len(grids)} images, patches per image mean {mean:.0f} (training "
    f"{TARGET:.0f}), min {min(patches)}, max {max(patches)}; merged tokens mean {mean / 4:.0f}"
)
