# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""H2D/D2H bandwidth: pageable vs Torch pin vs native unregistered vs native registered."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import torch

from deepspeed.accelerator import get_accelerator

ARMS = {
    "pageable": {
        "DS_PIN_MEMORY_BACKEND": "torch",
        "DS_PIN_MEMORY_REGISTER_DEVICE": "0"
    },
    "torch": {
        "DS_PIN_MEMORY_BACKEND": "torch",
        "DS_PIN_MEMORY_REGISTER_DEVICE": "1"
    },
    "native-unregistered": {
        "DS_PIN_MEMORY_BACKEND": "native",
        "DS_PIN_MEMORY_REGISTER_DEVICE": "0"
    },
    "native-registered": {
        "DS_PIN_MEMORY_BACKEND": "native",
        "DS_PIN_MEMORY_REGISTER_DEVICE": "1"
    },
}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=list(ARMS))
    parser.add_argument("--sizes-mib", type=int, nargs="+", default=[4, 64, 256])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--skip-native-register",
                        action="store_true",
                        help="Skip native-registered (XPU / no register_host_memory).")
    return parser.parse_args()


def _time_copy(accelerator, copy_fn, stream, warmup, iters):
    with accelerator.stream(stream):
        for _ in range(warmup):
            copy_fn()
        stream.synchronize()
        start = accelerator.Event(enable_timing=True)
        end = accelerator.Event(enable_timing=True)
        start.record(stream)
        for _ in range(iters):
            copy_fn()
        end.record(stream)
    stream.synchronize()
    return start.elapsed_time(end) / 1000.0 / iters


def _allocate_host(accelerator, numel, arm):
    raw = torch.empty(numel, dtype=torch.float32)
    if arm == "pageable":
        return raw
    if arm == "torch":
        return accelerator._torch_pin_memory(raw)
    return accelerator.pin_memory(raw, make_copy=False)


def _run_arm(args):
    for key, value in ARMS[args.arm].items():
        os.environ[key] = value

    accelerator = get_accelerator()
    if not accelerator.is_available():
        raise RuntimeError(f"No {accelerator.device_name()} device is available")
    accelerator.set_device(0)
    stream = accelerator.Stream()

    for size_mib in args.sizes_mib:
        num_bytes = size_mib * 1024 * 1024
        numel = num_bytes // torch.tensor([], dtype=torch.float32).element_size()
        host = _allocate_host(accelerator, numel, args.arm)
        device = torch.empty_like(host, device=accelerator.current_device_name())

        h2d_seconds = _time_copy(accelerator, lambda: device.copy_(host, non_blocking=True), stream, args.warmup,
                                 args.iters)
        d2h_seconds = _time_copy(accelerator, lambda: host.copy_(device, non_blocking=True), stream, args.warmup,
                                 args.iters)

        result = {
            "experiment": "h2d_d2h",
            "arm": args.arm,
            "size_mib": size_mib,
            "h2d_gbps": num_bytes / h2d_seconds / 1e9,
            "d2h_gbps": num_bytes / d2h_seconds / 1e9,
            "torch_is_pinned": bool(accelerator._torch_is_pinned(host)) if args.arm != "pageable" else False,
            "accelerator_is_pinned": bool(accelerator.is_pinned(host)) if args.arm != "pageable" else False,
        }
        print(f"RESULT={json.dumps(result, sort_keys=True)}", flush=True)
        if args.arm != "pageable":
            accelerator.unpin_memory(host)


def _arms_to_run(args):
    arms = list(ARMS)
    if args.skip_native_register:
        arms = [arm for arm in arms if arm != "native-registered"]
    return arms


def _run_all(args):
    results = []
    for arm in _arms_to_run(args):
        command = [
            sys.executable,
            os.path.abspath(__file__),
            "--arm",
            arm,
            "--sizes-mib",
            *(str(size) for size in args.sizes_mib),
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
        ]
        process = subprocess.run(command, check=True, text=True, capture_output=True)
        if process.stderr:
            print(process.stderr, file=sys.stderr, end="")
        for line in process.stdout.splitlines():
            print(line)
            if line.startswith("RESULT="):
                results.append(json.loads(line.removeprefix("RESULT=")))

    print("\narm,size_mib,h2d_gbps,d2h_gbps,torch_is_pinned,accelerator_is_pinned")
    for result in results:
        print(f"{result['arm']},{result['size_mib']},{result['h2d_gbps']:.2f},{result['d2h_gbps']:.2f},"
              f"{result['torch_is_pinned']},{result['accelerator_is_pinned']}")


if __name__ == "__main__":
    arguments = _parse_args()
    if arguments.arm:
        _run_arm(arguments)
    else:
        _run_all(arguments)
