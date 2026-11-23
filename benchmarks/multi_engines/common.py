# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Device-agnostic helpers for the multi-engine process-group checks.

Environment variables (all optional):
  DEVICE   accelerator type: cuda, xpu, hpu, npu, ... or cpu. Default: the current torch accelerator, else cpu.
  BACKEND  torch.distributed backend. Default: torch's default backend for DEVICE (nccl, xccl, hccl, gloo, ...).
  EAGER    1 binds the device at init_process_group(device_id=...), 0 leaves init lazy. Default 1 (DeepSpeed's CUDA default).
"""
import datetime
import json
import os
import threading
import time

import torch
import torch.distributed as dist

FALLBACK_BACKENDS = {"cuda": "nccl", "xpu": "xccl", "hpu": "hccl", "npu": "hccl", "cpu": "gloo"}


def rank():
    return int(os.environ.get("RANK", "0"))


def emit(case, **kw):
    """One JSON result record per line, prefixed with 'MER ' so summarize.py can find it."""
    kw.update(case=case, rank=rank(), t=round(time.time(), 3))
    print("MER " + json.dumps(kw, default=str), flush=True)


def watchdog(seconds, case):
    """Kill this process if the case hangs, and record that it did."""

    def _kill():
        emit(case, mark="hang_timeout", after_s=seconds)
        os._exit(3)

    timer = threading.Timer(seconds, _kill)
    timer.daemon = True
    timer.start()
    return timer


def device_type():
    if os.environ.get("DEVICE"):
        return os.environ["DEVICE"]
    acc = getattr(torch, "accelerator", None)
    if acc is not None and acc.is_available():
        return acc.current_accelerator().type
    return "cuda" if torch.cuda.is_available() else "cpu"


class Env:
    """The device, backend, and init mode for this process, plus device-agnostic helpers."""

    def __init__(self, pg_timeout_s=30):
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.type = device_type()
        self.on_cpu = self.type == "cpu"
        self.device = torch.device("cpu") if self.on_cpu else torch.device(self.type, self.local_rank)
        self.mod = None if self.on_cpu else getattr(torch, self.type)
        if not self.on_cpu:
            acc = getattr(torch, "accelerator", None)
            if acc is not None and hasattr(acc, "set_device_index"):
                acc.set_device_index(self.local_rank)
            else:
                self.mod.set_device(self.local_rank)
        self.backend = os.environ.get("BACKEND") or self._default_backend()
        self.eager = os.environ.get("EAGER", "1") == "1" and not self.on_cpu
        kw = {"device_id": self.device} if self.eager else {}
        dist.init_process_group(self.backend, timeout=datetime.timedelta(seconds=pg_timeout_s), **kw)

    def _default_backend(self):
        fn = getattr(dist, "get_default_backend_for_device", None)
        if fn is not None:
            try:
                return fn(self.device)
            except Exception:  # noqa: BLE001
                pass
        return FALLBACK_BACKENDS.get(self.type, "gloo")

    def sync(self):
        if not self.on_cpu:
            self.mod.synchronize()

    def free_mib(self):
        """Device-wide free memory, or None where the accelerator doesn't report it."""
        if self.on_cpu or not hasattr(self.mod, "mem_get_info"):
            return None
        self.sync()
        return round(self.mod.mem_get_info()[0] / 2**20, 1)

    def reserved_mib(self):
        if self.on_cpu or not hasattr(self.mod, "memory_reserved"):
            return None
        return round(self.mod.memory_reserved() / 2**20, 1)

    def backend_of(self, pg=None):
        g = pg if pg is not None else dist.distributed_c10d._get_default_group()
        try:
            return g._get_backend(self.device)
        except Exception:  # noqa: BLE001
            return None

    def error_state(self, pg=None):
        """The backend's communicator error state (e.g. SUCCESS, TIMEOUT), or why it is unavailable."""
        be = self.backend_of(pg)
        if be is None:
            return "unavailable (no backend object)"
        try:
            return str(be.get_error())
        except Exception as e:  # noqa: BLE001
            return f"unavailable ({type(e).__name__})"

    def info(self):
        be = self.backend_of()
        out = {"torch": torch.__version__, "world_size": self.world_size, "device": str(self.device),
               "backend": self.backend, "eager": self.eager,
               "backend_impl": type(be).__name__ if be is not None else None,
               "TORCH_DIST_USE_NCCL2": os.environ.get("TORCH_DIST_USE_NCCL2")}
        try:
            out["group_count"] = dist.distributed_c10d._world.group_count
        except Exception:  # noqa: BLE001
            pass
        return out


class Checker:
    """All-reduce checks with expected values; records every result."""

    def __init__(self, env, case):
        self.env, self.case, self.failures = env, case, []

    def group(self, label, g, ranks):
        """Size and rank-sum all-reduces over g; ranks are the expected members (global ranks)."""
        env = self.env
        t = torch.ones(16, device=env.device)
        dist.all_reduce(t, group=g)
        u = torch.full((16, ), float(env.rank), device=env.device)
        dist.all_reduce(u, group=g)
        env.sync()
        got = [t[0].item(), u[0].item()]
        expected = [float(len(ranks)), float(sum(ranks))]
        ok = got == expected
        if not ok:
            self.failures.append(label)
        emit(self.case, mark=f"check {label}", got=got, expected=expected, ok=ok)

    def created(self, label, g, ranks):
        """Log the group's name and torch's group counter, which must match across ranks."""
        W = dist.distributed_c10d._world
        name = None if g is None or g == dist.GroupMember.NON_GROUP_MEMBER else W.pg_names.get(g)
        emit(self.case, mark=f"created {label}", ranks=list(ranks), member=self.env.rank in ranks, name=name,
             group_count=getattr(W, "group_count", None))

    def summary(self):
        emit(self.case, mark="summary", result="ALL_OK" if not self.failures else "MISMATCH", failures=self.failures)
        return 0 if not self.failures else 1
