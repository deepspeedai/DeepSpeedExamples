# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Turn modal_bench.py grid results (bench/results/*.json) into Markdown tables.

    python bench/analyze.py bench/results/<run>.json [> table.md]

Per cell: median / p90 step time over the measured steps, tokens/s,
tokens/s/GPU, MFU (one FLOP formula for both frameworks), pipeline
efficiency E = tokens/s(PP=p) / (p x tokens/s(PP=1, same system and s)),
the textbook bubble (p-1)/(m+p-1), and NVML peak memory (max over GPUs).
"""

import json
import statistics
import sys
from collections import defaultdict

H100_BF16_DENSE = 989e12
L, H, Q, KV, FFN, V = 28, 1024, 2048, 1024, 3072, 151936


def _flops(s: int) -> float:
    """Training FLOPs per token, Megatron's convention: 3x the forward GEMMs,
    causal core attention counted at half."""
    per_layer_gemm = 2 * (H * (Q + 2 * KV) + Q * H) + 2 * 3 * H * FFN  # 31.46 MFLOP
    lm_head = 2 * H * V  # 311.16 MFLOP
    core_attention = 2 * 2 * Q * s / 2  # per layer
    return 3 * (L * per_layer_gemm + lm_head + L * core_attention)


def rows_per_microbatch(seq):
    """Every grid cell has 2048 tokens per microbatch."""
    return 4 if seq == 512 else 1


def summarize(records):
    out = []
    for r in records:
        c, res = r["cell"], r["result"]
        steps = res.get("step_s") or []
        gpus = c["pp"] - 1 + c.get("last_stage_gpus", 1)
        label = c["system"] + (f" ({c['tag']})" if c.get("tag") else "")
        row = dict(
            system=label,
            pp=c["pp"],
            m=c["m"],
            seq=c["seq"],
            gpus=gpus,
            rep=c.get("rep", 0),
            rc=r["rc"],
            peak_gb=max(r["nvml_peak_bytes"].values(), default=0) / 1e9,
        )
        if steps:
            med = statistics.median(steps)
            tokens = rows_per_microbatch(c["seq"]) * c["seq"] * c["m"]
            tps = tokens / med
            row.update(
                median_s=med,
                p90_s=sorted(steps)[int(0.9 * (len(steps) - 1))],
                tok_s=tps,
                tok_s_gpu=tps / gpus,
                mfu=_flops(c["seq"]) * tps / (gpus * H100_BF16_DENSE),
            )
        out.append(row)
    return out


def with_efficiency(rows):
    base = defaultdict(list)
    for r in rows:
        if r["pp"] == 1 and "tok_s" in r:
            base[(r["system"], r["seq"])].append(r["tok_s"])
    for r in rows:
        family = r["system"].split(" ")[0]
        # each system vs its own PP=1 baseline; transport only matters at PP>1,
        # so the RDT variants share the object-store PP=1 baseline
        family = {"R-rdt": "R-os", "R-rdt-fused": "R-os-fused"}.get(family, family)
        key = (family, r["seq"])
        if "tok_s" in r and base.get(key):
            r["eff"] = r["tok_s"] / (r["pp"] * statistics.median(base[key]))
        r["bubble_textbook"] = (r["pp"] - 1) / (r["m"] + r["pp"] - 1)
    return rows


def table(rows):
    hdr = (
        "| system | PP | GPUs | m | seq | step (median / p90) | tokens/s | tokens/s/GPU | MFU "
        "| efficiency E | textbook bubble | peak GB |"
    )
    lines = [hdr, "|" + "---|" * 12]
    for r in sorted(rows, key=lambda r: (r["seq"], r["pp"], r["m"], r["system"])):
        if "tok_s" not in r:
            lines.append(
                f"| {r['system']} | {r['pp']} | {r['gpus']} | {r['m']} | {r['seq']} "
                f"| FAILED rc={r['rc']} | | | | | | {r['peak_gb']:.1f} |"
            )
            continue
        eff = f"{r['eff']:.2f}" if "eff" in r else "–"
        lines.append(
            f"| {r['system']} | {r['pp']} | {r['gpus']} | {r['m']} | {r['seq']} "
            f"| {r['median_s'] * 1000:.0f} / {r['p90_s'] * 1000:.0f} ms | {r['tok_s']:,.0f} "
            f"| {r['tok_s_gpu']:,.0f} | {100 * r['mfu']:.1f}% | {eff} "
            f"| {r['bubble_textbook']:.2f} | {r['peak_gb']:.1f} |"
        )
    return "\n".join(lines)


def ratios(rows):
    """rdsp / Megatron throughput per (PP, m, seq)."""
    by = {(r["system"], r["pp"], r["m"], r["seq"]): r for r in rows if "tok_s" in r}
    lines = [
        "| PP | m | seq | R-os-fused / M-defaults | R-rdt-fused / M-defaults "
        "| R-os / M-plain | R-rdt / R-os |",
        "|---|---|---|---|---|---|---|",
    ]
    keys = sorted(
        {(pp, m, seq) for (_, pp, m, seq) in by if pp > 1}, key=lambda k: (k[2], k[0], k[1])
    )

    def ratio(num, den, pp, m, seq):
        a, b = by.get((num, pp, m, seq)), by.get((den, pp, m, seq))
        return f"{a['tok_s'] / b['tok_s']:.2f}" if a and b else "–"

    for pp, m, seq in keys:
        cols = [
            ratio(num, den, pp, m, seq)
            for num, den in (
                ("R-os-fused", "M-defaults"),
                ("R-rdt-fused", "M-defaults"),
                ("R-os", "M-plain"),
                ("R-rdt", "R-os"),
            )
        ]
        lines.append(f"| {pp} | {m} | {seq} | " + " | ".join(cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    records = []
    for path in sys.argv[1:]:
        with open(path) as f:
            records += json.load(f)
    rows = with_efficiency(summarize(records))
    print(table(rows))
    print()
    print(ratios(rows))
