# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""CPU-only: native pin vs Torch (Torch cannot pin without an accelerator)."""

from __future__ import annotations

import argparse
import os
import sys

_PIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PIN_ROOT not in sys.path:
    sys.path.insert(0, _PIN_ROOT)

from common import print_arm_result, print_driver_result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numel", type=int, default=1024 * 1024)
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    from deepspeed.accelerator import get_accelerator

    accel = get_accelerator()
    os.environ["DS_PIN_MEMORY_BACKEND"] = "native"
    host = torch.empty(args.numel, dtype=torch.float32)
    native = accel.pin_memory(host.clone(), make_copy=False)
    native_ok = bool(accel.is_pinned(native))
    accel.unpin_memory(native)

    os.environ["DS_PIN_MEMORY_BACKEND"] = "torch"
    torch_ok = None
    torch_error = None
    try:
        pinned = accel.pin_memory(torch.empty_like(host), make_copy=False)
        torch_ok = bool(accel.is_pinned(pinned))
    except Exception as exc:
        torch_error = type(exc).__name__ + ": " + str(exc)

    result = {
        "experiment": "cpu_pin",
        "accelerator": accel.device_name(),
        "native_is_pinned": native_ok,
        "torch_is_pinned": torch_ok,
        "torch_error": torch_error,
    }
    print_arm_result(result)
    print_driver_result(result)
    if not native_ok:
        raise SystemExit("native pin failed on this host")


if __name__ == "__main__":
    main()
