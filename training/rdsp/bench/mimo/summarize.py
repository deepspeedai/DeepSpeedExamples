# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Results table from bench/mimo/runs.sh logs: one row per Megatron / rdsp
pair, median step time over steps 6-50 (the first five compile and warm up).

Tokens/s is rows x sequence length / median step time (64 x 2048 in every
bench/mimo run); per GPU divides by the layout's GPU count.

    python summarize.py ~/runs --price 5.32
"""

import argparse
import re
import statistics
from pathlib import Path

PAIRS = (  # (layout, GPUs, Megatron run or None for rdsp only, rdsp run)
    ("vision 1 GPU + language TP2xDP2", 5, "noncoloc-mimo", "noncoloc-rdsp"),
    ("vision DP2 + language TP2xDP2", 6, "vision2-mimo", "vision2-rdsp"),
    ("vision 1 GPU + language TP2xPP2", 5, "langpp2-mimo", "langpp2-rdsp"),
    ("vision 1 GPU + language TP4", 5, "langtp4-mimo", "langtp4-rdsp"),
    ("TP2xPP2xDP2", 8, "shared-megatron", "shared-rdsp"),
    ("TP4xPP2xDP1", 8, "tp4pp2-megatron", "tp4pp2-rdsp"),
    ("colocated vision + TP2xPP2xDP2", 8, None, "coloc-rdsp"),
)
_MEGATRON = re.compile(r"iteration\s+(\d+)/\s*\d+ .*?elapsed time per iteration \(ms\): ([\d.]+)")
_RDSP = re.compile(r"^step (\d+) loss \S+ +(\d+) ms", re.M)


def parse(path: Path) -> dict:
    """{step: ms}, steps counted from 0 (Megatron's iteration 1 is step 0, as in rdsp)."""
    text = path.read_text(errors="replace")
    ms = {int(m[1]) - 1: float(m[2]) for m in _MEGATRON.finditer(text)}
    ms.update({int(m[1]): float(m[2]) for m in _RDSP.finditer(text)})
    return ms


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("logs")
    p.add_argument("--price", type=float, default=5.32, help="8-GPU node, $/hour")
    p.add_argument("--first", type=int, default=5, help="warm-up steps left out")
    p.add_argument("--rows", type=int, default=64, help="rows per step")
    p.add_argument("--seq", type=int, default=2048, help="sequence length")
    args = p.parse_args(argv)
    logs = Path(args.logs)
    print(
        f"{'layout':34} {'GPUs':>4} {'Megatron s':>10} {'rdsp s':>7} {'step':>6} "
        f"{'Megatron tok/s':>14} {'rdsp tok/s':>10} {'rdsp $/M tok':>12} "
        f"{'Megatron tok/s/GPU':>18} {'rdsp tok/s/GPU':>14}"
    )
    for layout, gpus, megatron, rdsp in PAIRS:
        if not (logs / f"{rdsp}.log").exists() or (
            megatron and not (logs / f"{megatron}.log").exists()
        ):
            continue
        r_ms = parse(logs / f"{rdsp}.log")
        m_ms = parse(logs / f"{megatron}.log") if megatron else None
        timed = [s for s in sorted(r_ms) if s >= args.first and (m_ms is None or s in m_ms)]
        if not timed:
            continue
        tokens = args.rows * args.seq
        r_s = statistics.median(r_ms[s] for s in timed) / 1e3
        r_tok = tokens / r_s
        cost = args.price * gpus / 8 / 3600 / r_tok * 1e6
        if m_ms is None:
            m_col, step, m_tok, m_gpu = "-", "-", "-", "-"
        else:
            m_s = statistics.median(m_ms[s] for s in timed) / 1e3
            m_col, step = f"{m_s:.2f}", f"{r_s / m_s - 1:+.0%}"
            tok = tokens / m_s
            m_tok, m_gpu = f"{tok:.0f}", f"{tok / gpus:.0f}"
        print(
            f"{layout:34} {gpus:>4} {m_col:>10} {r_s:>7.2f} {step:>6} "
            f"{m_tok:>14} {r_tok:>10.0f} {cost:>12.3f} {m_gpu:>18} {r_tok / gpus:>14.0f}"
        )


if __name__ == "__main__":
    main()
