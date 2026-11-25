# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Stage builders and engine factories that rdsp_bench.py injects into rdsp.

Imported inside the stage actors (rdsp_bench.py puts this directory on their
PYTHONPATH). The engine proxies wrap rdsp's own DeepSpeed engine factory
unmodified, so timing and profiling add no code to the measured path beyond
the proxy calls.
"""

import json
import os
import time

import torch

from ray_deepspeed_pipeline.deepspeed_adapter import _deepspeed_engine_factory
from ray_deepspeed_pipeline.partition import build_causal_lm_stage

OUT = os.environ.get("RDSP_BENCH_DIR", "/tmp/rdsp_bench")
WARMUP = int(os.environ.get("RDSP_BENCH_WARMUP", "10"))


def tagged_stage_builder(model, start, stop, names):
    m = build_causal_lm_stage(model, start, stop, names)
    m.bench_tag = f"blocks{start:02d}-{stop:02d}"
    return m


class TimedEngine:
    """Engine proxy: CUDA-event timing and NVTX ranges per forward, backward
    and optimizer step. One host sync per optimizer step; one JSON line per
    step in OUT. Peak-memory stats reset after WARMUP steps."""

    def __init__(self, engine, tag):
        self._e, self._tag, self._ops, self._step = engine, tag, [], 0
        os.makedirs(OUT, exist_ok=True)
        self._f = open(f"{OUT}/{tag}.rank{os.environ.get('RANK', '0')}.jsonl", "a")

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._e, name)

    def _timed(self, kind, fn, *args):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t = time.monotonic_ns()
        torch.cuda.nvtx.range_push(f"{self._tag}.{kind}")
        s.record()
        out = fn(*args)
        e.record()
        torch.cuda.nvtx.range_pop()
        self._ops.append((kind, t, s, e))
        return out

    def __call__(self, x, **kw):
        return self._timed("fwd", lambda t: self._e(t, **kw), x)

    def backward(self, loss):
        return self._timed("bwd", self._e.backward, loss)

    def step(self):
        out = self._timed("opt", self._e.step)
        torch.cuda.synchronize()
        self._f.write(
            json.dumps(
                {
                    "tag": self._tag,
                    "pid": os.getpid(),
                    "step": self._step,
                    "ops": [(k, t, s.elapsed_time(e)) for k, t, s, e in self._ops],
                    "max_alloc": torch.cuda.max_memory_allocated(),
                    "max_reserved": torch.cuda.max_memory_reserved(),
                }
            )
            + "\n"
        )
        self._f.flush()
        self._ops.clear()
        self._step += 1
        if self._step == WARMUP:
            torch.cuda.reset_peak_memory_stats()
        return out


def timed_engine_factory(stage_module, ds_config):
    return TimedEngine(
        _deepspeed_engine_factory(stage_module, ds_config),
        getattr(stage_module, "bench_tag", "stage"),
    )


# Fused kernels, to match Megatron's default fusions. Liger replaces Qwen3's
# RMSNorm and SwiGLU MLP classes and its rotary function. The class swaps
# travel with the pickled stage modules, but the rotary patch is a module-level
# function in transformers, so every stage worker applies it too.


def apply_fusions():
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3

    apply_liger_kernel_to_qwen3(
        rope=True, rms_norm=True, swiglu=True, cross_entropy=False, fused_linear_cross_entropy=False
    )


def fused_engine_factory(stage_module, ds_config):
    apply_fusions()
    return _deepspeed_engine_factory(stage_module, ds_config)


def fused_timed_engine_factory(stage_module, ds_config):
    apply_fusions()
    return timed_engine_factory(stage_module, ds_config)


# Kernel profile: same tool and window as megatron_hf_loop.py --prof_out.
PROF_START, PROF_STOP = 20, 30  # optimizer steps [20, 30): 10 steady-state steps


def kernel_table(prof):
    """{kernel name: [calls, total device us]} from a torch.profiler run."""
    out = {}
    for e in prof.key_averages():
        t = getattr(e, "self_device_time_total", None)
        if t is None:
            t = getattr(e, "self_cuda_time_total", 0)
        if t and str(getattr(e, "device_type", "")).endswith("CUDA"):
            out[e.key] = [e.count, t]
    return out


class ProfiledEngine:
    """Engine proxy: torch.profiler around optimizer steps [PROF_START,
    PROF_STOP); writes the kernel table to OUT/prof_<tag>.json."""

    def __init__(self, engine, tag):
        self._e, self._tag, self._n, self._prof, self._t0 = engine, tag, 0, None, 0.0

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._e, name)

    def __call__(self, x, **kw):
        return self._e(x, **kw)

    def backward(self, loss):
        return self._e.backward(loss)

    def step(self):
        out = self._e.step()
        self._n += 1
        if self._n == PROF_START:
            torch.cuda.synchronize()
            self._prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            )
            self._prof.__enter__()
            self._t0 = time.perf_counter()
        elif self._n == PROF_STOP and self._prof is not None:
            torch.cuda.synchronize()
            wall = time.perf_counter() - self._t0
            self._prof.__exit__(None, None, None)
            os.makedirs(OUT, exist_ok=True)
            with open(f"{OUT}/prof_{self._tag}.json", "w") as f:
                json.dump(
                    {
                        "steps": PROF_STOP - PROF_START,
                        "wall_s": wall,
                        "kernels": kernel_table(self._prof),
                    },
                    f,
                )
            self._prof = None
        return out


def fused_prof_engine_factory(stage_module, ds_config):
    apply_fusions()
    return ProfiledEngine(
        _deepspeed_engine_factory(stage_module, ds_config),
        getattr(stage_module, "bench_tag", "stage"),
    )
