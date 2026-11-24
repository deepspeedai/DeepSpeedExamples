# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Per-GPU time per training step from an Nsight Systems trace: how long each
GPU computes, communicates and sits idle. The same table for Megatron and rdsp,
so their difference is visible kernel by kernel.

    nsys export --type sqlite -o run.sqlite run.nsys-rep
    python nsys_breakdown.py run.sqlite --step-ms 7890

The window is the span of all kernels in the trace (runs.sh profiles a fixed
window of steady-state steps); --step-ms (the run's median step time) turns
its totals into per-step numbers. Busy time is the union of a GPU's kernel
intervals, so communication that overlaps compute is not counted twice;
category columns are each category's own kernel time. A GPU running the
decoder's linear-attention kernels is "language", any other "vision".
"""

import argparse
import re
import sqlite3

CATEGORIES = (  # first match wins
    ("all-reduce", r"nccl.*AllReduce"),
    ("reduce-scatter", r"nccl.*ReduceScatter"),
    ("all-gather", r"nccl.*AllGather"),
    ("send/recv", r"nccl.*(SendRecv|Send|Recv)"),
    ("other comm", r"nccl"),
    ("linear attention", r"gated_delta|delta_rule|causal_conv1d"),
    ("attention", r"flash|fmha|attention|sdpa"),
    ("matmul", r"gemm|xmma|nvjet|cutlass|cublas|wgmma"),
)
COLUMNS = [name for name, _ in CATEGORIES] + ["other"]


def category(name: str) -> str:
    for label, pattern in CATEGORIES:
        if re.search(pattern, name, re.IGNORECASE):
            return label
    return "other"


def _union(intervals) -> int:
    total, end = 0, None
    for s, e in sorted(intervals):
        if end is None or s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def breakdown(path: str, step_ms: float) -> list[dict]:
    """One dict per GPU: role, steps in the window, and per step: busy_ms,
    idle_ms, kernels, and each category's ms."""
    db = sqlite3.connect(path)
    rows = db.execute(
        "SELECT k.deviceId, k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id = k.shortName"
    ).fetchall()
    db.close()
    lo, hi = min(r[1] for r in rows), max(r[2] for r in rows)
    steps = (hi - lo) / 1e6 / step_ms
    out = []
    for device in sorted({r[0] for r in rows}):
        mine = [r for r in rows if r[0] == device]
        per = {c: 0 for c in COLUMNS}
        for _, s, e, name in mine:
            per[category(name)] += e - s
        busy = _union((s, e) for _, s, e, _ in mine)
        row = {
            "device": device,
            "steps": steps,
            "role": "language" if per["linear attention"] else "vision",
            "busy_ms": busy / 1e6 / steps,
            "idle_ms": ((hi - lo) - busy) / 1e6 / steps,
            "kernels": len(mine) / steps,
        }
        row.update({c: v / 1e6 / steps for c, v in per.items()})
        out.append(row)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sqlite")
    p.add_argument("--step-ms", type=float, required=True, help="median step time of the run")
    args = p.parse_args(argv)
    rows = breakdown(args.sqlite, args.step_ms)
    print(
        f"{'gpu':>3} {'role':9} {'busy':>7} {'idle':>7} {'kernels':>8} "
        + " ".join(f"{c:>16}" for c in COLUMNS)
        + "   (ms per step)"
    )
    for r in rows:
        print(
            f"{r['device']:>3} {r['role']:9} {r['busy_ms']:>7.0f} {r['idle_ms']:>7.0f} "
            f"{r['kernels']:>8.0f} " + " ".join(f"{r[c]:>16.0f}" for c in COLUMNS)
        )
    print(f"window: {rows[0]['steps']:.1f} steps")


if __name__ == "__main__":
    main()
