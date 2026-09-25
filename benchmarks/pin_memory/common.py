# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Shared helpers for pin_memory experiment drivers (subprocess arms, JSON lines)."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys


def free_port():
    # A stale listener from an interrupted rank makes the next init hang in a
    # collective, so always rendezvous on a fresh ephemeral port.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def dist_env():
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(free_port()),
        RANK="0",
        WORLD_SIZE="1",
        LOCAL_RANK="0",
    )


def patch_cpp_extension_drop_cxx17():
    # torch-nightly requires C++20; SYCL toolchain flags may carry -std=c++17,
    # which (appearing last) downgrades the dialect and breaks torch headers.
    import torch.utils.cpp_extension as cpp_ext

    orig_load = cpp_ext.load

    def load_without_cxx17(*args, **kwargs):
        for key in ("extra_cflags", "extra_cxxflags"):
            if kwargs.get(key):
                kwargs[key] = [flag for flag in kwargs[key] if flag != "-std=c++17"]
        return orig_load(*args, **kwargs)

    cpp_ext.load = load_without_cxx17


def print_arm_result(result):
    print("ARMRESULT=" + json.dumps(result), flush=True)


def print_driver_result(summary):
    print("DRIVERRESULT=" + json.dumps(summary), flush=True)


def parse_arm_result(stdout):
    for line in stdout.splitlines():
        if line.startswith("ARMRESULT="):
            return json.loads(line[len("ARMRESULT="):])
    return None


def run_arm_subprocess(script_path, extra_args, env=None):
    command = [sys.executable, os.path.abspath(script_path), *extra_args]
    print(f"[driver] {' '.join(command)}", flush=True)
    proc = subprocess.run(command, env=env or os.environ.copy(), capture_output=True, text=True)
    arm = parse_arm_result(proc.stdout)
    if arm is None:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"arm produced no ARMRESULT (rc={proc.returncode})")
    return arm
