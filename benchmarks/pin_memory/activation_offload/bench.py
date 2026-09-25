# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Activation / checkpoint hidden-state CPU offload: pin vs pageable with async on.

Holds use_streams=True. Compares use_pin_memory True vs False. Do not treat
this as DeepSpeed #8282's async-vs-blocking table.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_PIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PIN_ROOT not in sys.path:
    sys.path.insert(0, _PIN_ROOT)

from common import dist_env, patch_cpp_extension_drop_cxx17, print_arm_result, print_driver_result, run_arm_subprocess


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--pin", type=int, default=None, help="internal: use_pin_memory 0/1")
    return parser.parse_args()


def run_arm(args):
    dist_env()
    patch_cpp_extension_drop_cxx17()

    import torch
    from torch.utils.checkpoint import checkpoint

    from deepspeed.accelerator import get_accelerator
    from deepspeed.runtime.activation_checkpointing.offload_activations import CheckpointHiddenStatesOffload

    accelerator = get_accelerator()
    if not accelerator.is_available():
        raise RuntimeError(f"No {accelerator.device_name()} device is available")
    accelerator.set_device(0)
    device = accelerator.current_device_name()

    class Block(torch.nn.Module):

        def __init__(self, hidden):
            super().__init__()
            self.fc1 = torch.nn.Linear(hidden, 4 * hidden)
            self.fc2 = torch.nn.Linear(4 * hidden, hidden)

        def forward(self, hidden_states):
            return self.fc2(torch.nn.functional.gelu(self.fc1(hidden_states)))

    class Net(torch.nn.Module):

        def __init__(self, hidden, layers):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block(hidden) for _ in range(layers)])

        def forward(self, hidden_states, offload):
            x = hidden_states
            for block in self.blocks:
                offload.mark(x)
                x = x + checkpoint(block, x, use_reentrant=False)
            return x.sum()

    model = Net(args.hidden, args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(args.batch, args.seq, args.hidden, device=device, requires_grad=True)

    # Async side stream stays on; pin vs pageable is the only axis.
    offload = CheckpointHiddenStatesOffload(use_pin_memory=bool(args.pin),
                                            use_streams=True,
                                            min_offload_bytes=0,
                                            keep_last_count=1)

    step_times = []
    for step in range(args.warmup + args.steps):
        accelerator.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with offload:
            loss = model(x, offload)
            loss.backward()
        opt.step()
        accelerator.synchronize()
        if step >= args.warmup:
            step_times.append(time.perf_counter() - t0)
        offload.reset()

    def _peak():
        try:
            return round(torch.get_device_module(device).max_memory_allocated() / 1e9, 2)
        except Exception:
            return None

    print_arm_result({
        "experiment": "activation_offload",
        "use_pin_memory": bool(args.pin),
        "use_streams": True,
        "device": accelerator.device_name(),
        "hidden": args.hidden,
        "layers": args.layers,
        "batch": args.batch,
        "seq": args.seq,
        "steps": len(step_times),
        "step_avg_s": sum(step_times) / len(step_times),
        "step_min_s": min(step_times),
        "gpu_peak_gb": _peak(),
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

    pageable = results["pageable"]
    pinned = results["pinned"]
    print("\n================ Activation offload (async on) ================")
    print(f"device: {pinned['device']}  hidden: {pinned['hidden']}  layers: {pinned['layers']}  "
          f"batch: {pinned['batch']} seq: {pinned['seq']}")
    print(f"{'arm':<22}{'avg step (s)':>14}{'min step (s)':>14}{'GPU peak (GB)':>16}")
    for name in ("pageable", "pinned"):
        row = results[name]
        print(f"{name:<22}{row['step_avg_s']:>14.3f}{row['step_min_s']:>14.3f}"
              f"{(row['gpu_peak_gb'] or 0):>16.2f}")
    speedup = pageable["step_avg_s"] / pinned["step_avg_s"]
    print()
    print(f"pin vs pageable step-time ratio: {speedup:.2f}x (async held on)")
    print_driver_result({"pageable": pageable, "pinned": pinned, "speedup": speedup})


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.pin is None:
        run_driver(parsed)
    else:
        run_arm(parsed)
