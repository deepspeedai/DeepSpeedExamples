# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""DeepCompile activation offload pin knob (compile.offload_activation_pin_memory)."""

from __future__ import annotations

import argparse
import os
import sys

_PIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PIN_ROOT not in sys.path:
    sys.path.insert(0, _PIN_ROOT)

from common import dist_env, patch_cpp_extension_drop_cxx17, print_arm_result, print_driver_result, run_arm_subprocess


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--pin", type=int, default=None)
    return parser.parse_args()


def run_arm(args):
    dist_env()
    patch_cpp_extension_drop_cxx17()
    import time
    import torch
    import deepspeed

    class Net(torch.nn.Module):

        def __init__(self, hidden, layers):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(hidden, hidden) for _ in range(layers)])

        def forward(self, x):
            for layer in self.layers:
                x = torch.nn.functional.gelu(layer(x))
            return x.sum()

    model = Net(args.hidden, args.layers)
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch,
        "gradient_accumulation_steps": 1,
        "bf16": {
            "enabled": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4
            }
        },
        "zero_optimization": {
            "stage": 3
        },
        "compile": {
            "enabled": True,
            "offload_activation": True,
            "offload_activation_pin_memory": bool(args.pin),
        },
    }
    engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
    dev = engine.device
    x = torch.randn(args.batch, args.seq, args.hidden, device=dev)
    times = []
    for step in range(args.warmup + args.steps):
        t0 = time.perf_counter()
        loss = engine(x)
        engine.backward(loss)
        engine.step()
        if step >= args.warmup:
            times.append(time.perf_counter() - t0)
    print_arm_result({
        "experiment": "deepcompile_activation",
        "offload_activation_pin_memory": bool(args.pin),
        "device": str(dev),
        "step_avg_s": sum(times) / len(times),
    })


def run_driver(args):
    script = os.path.abspath(__file__)
    base = [
        "--hidden",
        str(args.hidden),
        "--layers",
        str(args.layers),
        "--batch",
        str(args.batch),
        "--seq",
        str(args.seq),
        "--steps",
        str(args.steps),
        "--warmup",
        str(args.warmup),
    ]
    results = {}
    for name, pin in (("pageable", 0), ("pinned", 1)):
        results[name] = run_arm_subprocess(script, base + ["--pin", str(pin)])
    print_driver_result(results)


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.pin is None:
        run_driver(parsed)
    else:
        run_arm(parsed)
