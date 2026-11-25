#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""summarize.py <result.json>...: markdown tables from enc_bench.py results (files without
results, e.g. grids.json, are skipped)."""

import json
import sys

PARTS = (
    "rotary",
    "core_attn",
    "linear",
    "norm",
    "attn_other",
    "mlp_other",
    "layer_other",
    "outside",
    "outside_encoder",
    "backward",
)


def f(x, nd=1):
    return "" if x is None else f"{x:,.{nd}f}"


runs = []
for path in sys.argv[1:]:
    data = json.load(open(path))
    if isinstance(data, dict) and "results" in data:
        runs.append(data)

print("## Environments\n")
for r in runs:
    label = r["results"][0]["label"] if r["results"] else r["args"]["label"]
    print(f"- {label}: {json.dumps(r['env'])}")

for mode in ("fwd", "fwdbwd"):
    print(f"\n## {mode}: per image\n")
    print(
        "| Encoder | Images per call | Time (ms) | GPU busy (ms) | GPU share | Kernels | Syncs | "
        "Traced time (ms) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in runs:
        for x in r["results"]:
            if x["mode"] != mode:
                continue
            busy = x.get("gpu_busy_ms_per_image")
            share = f"{busy / x['ms_per_image']:.0%}" if busy else ""
            print(
                f"| {x['label']} | {x['images_per_call']} | {f(x['ms_per_image'])} | {f(busy)} | "
                f"{share} | {f(x.get('kernels_per_image'), 0)} | "
                f"{f(x.get('syncs_per_image'), 0)} | {f(x.get('traced_ms_per_image'))} |"
            )

for what, key, nd in (
    ("kernels", "kernels_by_part_per_image", 0),
    ("GPU ms", "gpu_ms_by_part_per_image", 2),
    ("CPU ms (traced, forward thread, exclusive)", "cpu_ms_by_part_per_image", 2),
):
    print(f"\n## fwd, 1 image per call: {what} per image, by part\n")
    print("| Part | " + " | ".join(r["results"][0]["label"] for r in runs if r["results"]) + " |")
    print("|---|" + "---|" * len(runs))
    for part in PARTS:
        cells = []
        for r in runs:
            x = next(
                (x for x in r["results"] if x["mode"] == "fwd" and x["images_per_call"] == 1), {}
            )
            cells.append(f(x.get(key, {}).get(part), nd))
        if any(cells):
            print(f"| {part} | " + " | ".join(cells) + " |")
