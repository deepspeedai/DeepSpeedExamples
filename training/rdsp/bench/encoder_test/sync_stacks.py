#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""sync_stacks.py --grids grids.json: which Python lines in Megatron-Bridge's vision encoder make
the CPU wait for the GPU. One image per call, forward; torch.cuda.set_sync_debug_mode("warn")
flags every synchronizing operation, and the warning's Python stack names the line.
Prints each distinct call site (the innermost frames outside torch) with its count per image."""

import argparse
import collections
import json
import traceback
import warnings

import enc_bench
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--grids", required=True)
ap.add_argument("--config", default="Qwen/Qwen3.5-27B")
ap.add_argument("--images", type=int, default=4)
args = ap.parse_args()

enc, hf_vision = enc_bench.build_bridge(args.config)
grids = json.load(open(args.grids))[: args.images]
calls = enc_bench.make_calls("bridge", enc, hf_vision, grids, 1, "cuda")
for c in calls[:2]:
    enc_bench.run_call("bridge", enc, c, False)
torch.cuda.synchronize()

sites = collections.Counter()


def record(message, category, filename, lineno, file=None, line=None):
    frames = [
        f
        for f in traceback.extract_stack()[:-1]
        if "/torch/" not in f.filename
        and "warnings" not in f.filename
        and "sync_stacks.py" not in f.filename
        and "enc_bench.py" not in f.filename
    ]
    site = " <- ".join(
        f"{f.filename.split('site-packages/')[-1].split('dist-packages/')[-1]}:{f.lineno} {f.line}"
        for f in reversed(frames[-4:])
    )
    sites[site] += 1


warnings.showwarning = record
warnings.simplefilter("always")
torch.cuda.set_sync_debug_mode("warn")
for c in calls:
    enc_bench.run_call("bridge", enc, c, False)
torch.cuda.set_sync_debug_mode("default")
torch.cuda.synchronize()
total = sum(sites.values())
print(f"{total / len(calls):.1f} synchronizing operations per image\n")
for site, n in sites.most_common(20):
    print(f"{n / len(calls):6.1f} per image\n    " + site.replace(" <- ", "\n      <- ") + "\n")
