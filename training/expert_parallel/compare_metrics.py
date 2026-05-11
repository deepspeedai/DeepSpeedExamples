"""Compare AutoEP and ZeRO-3 leaf CSV metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BYTES_PER_GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare AutoEP and ZeRO-3 leaf metrics")
    parser.add_argument("--autoep_csv", required=True)
    parser.add_argument("--zero3_leaf_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--autoep_label", default="AutoEP + ZeRO-1")
    parser.add_argument("--zero3_leaf_label", default="HF + ZeRO-3 leaf")
    parser.add_argument("--warmup_steps", type=int, default=5)
    return parser.parse_args()


def load_rows(path: str, warmup_steps: int) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        rows = [row for row in csv.DictReader(f) if int(row["step"]) >= warmup_steps]
    if not rows:
        raise ValueError(f"No rows at or after warmup step {warmup_steps}: {path}")
    return rows


def align_rows(
    autoep_rows: list[dict[str, str]],
    zero3_rows: list[dict[str, str]],
) -> tuple[list[int], list[dict[str, str]], list[dict[str, str]]]:
    autoep_by_step = {int(row["step"]): row for row in autoep_rows}
    zero3_by_step = {int(row["step"]): row for row in zero3_rows}
    steps = sorted(set(autoep_by_step) & set(zero3_by_step))
    if not steps:
        raise ValueError("AutoEP and ZeRO-3 metrics do not share any post-warmup steps")
    return steps, [autoep_by_step[step] for step in steps], [zero3_by_step[step] for step in steps]


def avg(rows: list[dict[str, str]], column: str) -> float:
    return mean(float(row[column]) for row in rows)


def max_int(rows: list[dict[str, str]], column: str) -> int:
    return max(int(row[column]) for row in rows)


def write_json(path: str, payload: dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def save_loss_curve(
    steps: list[int],
    autoep_rows: list[dict[str, str]],
    zero3_rows: list[dict[str, str]],
    autoep_label: str,
    zero3_label: str,
    out_dir: str,
) -> str:
    path = os.path.join(out_dir, "loss_curve.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, [float(row["loss"]) for row in autoep_rows], label=autoep_label)
    ax.plot(steps, [float(row["loss"]) for row in zero3_rows], label=zero3_label)
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_bar_chart(
    values: list[float],
    labels: list[str],
    ylabel: str,
    title: str,
    path: str,
    value_format: str,
) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            value_format.format(bar.get_height()),
            ha="center",
            va="bottom",
        )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    autoep_rows = load_rows(args.autoep_csv, args.warmup_steps)
    zero3_rows = load_rows(args.zero3_leaf_csv, args.warmup_steps)
    steps, autoep_aligned, zero3_aligned = align_rows(autoep_rows, zero3_rows)

    autoep_loss = avg(autoep_aligned, "loss")
    zero3_loss = avg(zero3_aligned, "loss")
    autoep_tps = avg(autoep_aligned, "global_tokens_per_sec")
    zero3_tps = avg(zero3_aligned, "global_tokens_per_sec")
    autoep_peak_mem = max_int(autoep_aligned, "cuda_peak_memory_allocated_bytes")
    zero3_peak_mem = max_int(zero3_aligned, "cuda_peak_memory_allocated_bytes")

    labels = [args.autoep_label, args.zero3_leaf_label]
    plots = {
        "loss_curve": save_loss_curve(
            steps,
            autoep_aligned,
            zero3_aligned,
            args.autoep_label,
            args.zero3_leaf_label,
            args.out_dir,
        ),
        "peak_memory_bar": save_bar_chart(
            [autoep_peak_mem / BYTES_PER_GIB, zero3_peak_mem / BYTES_PER_GIB],
            labels,
            "Peak Memory (GiB)",
            "Peak GPU Memory Comparison",
            os.path.join(args.out_dir, "peak_memory_bar.png"),
            "{:.2f}",
        ),
        "throughput_bar": save_bar_chart(
            [autoep_tps, zero3_tps],
            labels,
            "Tokens/sec",
            "Average Throughput Comparison",
            os.path.join(args.out_dir, "throughput_bar.png"),
            "{:.0f}",
        ),
    }

    summary = {
        "aligned_steps": len(steps),
        "loss": {
            "autoep_mean": autoep_loss,
            "zero3_leaf_mean": zero3_loss,
            "mean_abs_diff": abs(autoep_loss - zero3_loss),
        },
        "throughput": {
            "autoep_tokens_per_sec": autoep_tps,
            "zero3_leaf_tokens_per_sec": zero3_tps,
            "ratio": autoep_tps / zero3_tps if zero3_tps else None,
        },
        "peak_memory": {
            "autoep_bytes": autoep_peak_mem,
            "zero3_leaf_bytes": zero3_peak_mem,
            "autoep_gib": autoep_peak_mem / BYTES_PER_GIB,
            "zero3_leaf_gib": zero3_peak_mem / BYTES_PER_GIB,
            "ratio": autoep_peak_mem / zero3_peak_mem if zero3_peak_mem else None,
        },
        "plots": plots,
    }
    write_json(args.out_json, summary)

    print("\n=== Comparison Summary ===")
    print(f"Aligned steps: {len(steps)}")
    print(f"Mean loss: AutoEP={autoep_loss}, ZeRO-3={zero3_loss}")
    print(f"Mean abs diff (loss): {summary['loss']['mean_abs_diff']}")
    print(f"Peak memory ratio (AutoEP / ZeRO-3): {summary['peak_memory']['ratio']}")
    print(f"Throughput ratio (AutoEP / ZeRO-3): {summary['throughput']['ratio']}")
    print(f"Summary written to: {args.out_json}")


if __name__ == "__main__":
    main()
