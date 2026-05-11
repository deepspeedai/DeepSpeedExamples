"""Compare AutoEP and ZeRO-3 leaf CSV metrics.

This script intentionally consumes only the public metrics CSV columns emitted by
train.py: loss, throughput, and CUDA memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare AutoEP and ZeRO-3 leaf metrics")
    parser.add_argument("--autoep_csv", required=True)
    parser.add_argument("--zero3_leaf_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--autoep_label", default="AutoEP + ZeRO-1")
    parser.add_argument("--zero3_leaf_label", default="HF + ZeRO-3 leaf")
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--max_mean_abs_diff", type=float, default=None)
    parser.add_argument("--min_post_warmup_steps", type=int, default=10)
    return parser.parse_args()


def load_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pearson_correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    numerator = sum(a * b for a, b in zip(dx, dy))
    denom_x = math.sqrt(sum(a * a for a in dx))
    denom_y = math.sqrt(sum(b * b for b in dy))
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / (denom_x * denom_y)


def metric_parity(
    autoep_steps: dict[int, dict[str, str]],
    zero3_steps: dict[int, dict[str, str]],
    aligned_steps: list[int],
    metric_name: str,
    min_corr_steps: int,
) -> dict[str, Any]:
    a_values = [float(autoep_steps[step][metric_name]) for step in aligned_steps]
    z_values = [float(zero3_steps[step][metric_name]) for step in aligned_steps]
    if not aligned_steps:
        return {
            "recorded": False,
            "mean_abs_diff": None,
            "max_abs_diff": None,
            "pearson_correlation": None,
            "num_aligned_steps": 0,
        }
    abs_diffs = [abs(a - z) for a, z in zip(a_values, z_values)]
    return {
        "recorded": True,
        "mean_abs_diff": sum(abs_diffs) / len(abs_diffs),
        "max_abs_diff": max(abs_diffs),
        "pearson_correlation": (
            pearson_correlation(a_values, z_values)
            if len(aligned_steps) >= min_corr_steps
            else None
        ),
        "num_aligned_steps": len(aligned_steps),
    }


def try_plot(
    autoep_rows: list[dict[str, str]],
    zero3_rows: list[dict[str, str]],
    out_dir: str,
    autoep_label: str,
    zero3_leaf_label: str,
) -> dict[str, str | None]:
    plots = {
        "loss_curve": None,
        "peak_memory_bar": None,
        "throughput_bar": None,
    }
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available; skipping plots.")
        return plots

    os.makedirs(out_dir, exist_ok=True)

    def plot_curve(metric: str, ylabel: str, title: str, filename: str) -> str:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            [int(row["step"]) for row in autoep_rows],
            [float(row[metric]) for row in autoep_rows],
            label=autoep_label,
            marker="o",
            markersize=3,
        )
        ax.plot(
            [int(row["step"]) for row in zero3_rows],
            [float(row[metric]) for row in zero3_rows],
            label=zero3_leaf_label,
            marker="s",
            markersize=3,
        )
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    try:
        plots["loss_curve"] = plot_curve("loss", "Loss", "Loss Curve Comparison", "loss_curve.png")
    except Exception as exc:
        print(f"WARNING: loss curve plot failed: {exc}")

    try:
        a_peak = max(int(row["cuda_peak_memory_allocated_bytes"]) for row in autoep_rows)
        z_peak = max(int(row["cuda_peak_memory_allocated_bytes"]) for row in zero3_rows)
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar([autoep_label, zero3_leaf_label], [a_peak / 1e9, z_peak / 1e9])
        ax.set_ylabel("Peak Memory (GB)")
        ax.set_title("Peak GPU Memory Comparison")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{bar.get_height():.2f}", ha="center", va="bottom")
        path = os.path.join(out_dir, "peak_memory_bar.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots["peak_memory_bar"] = path
    except Exception as exc:
        print(f"WARNING: memory plot failed: {exc}")

    try:
        a_tps = sum(float(row["global_tokens_per_sec"]) for row in autoep_rows) / len(autoep_rows)
        z_tps = sum(float(row["global_tokens_per_sec"]) for row in zero3_rows) / len(zero3_rows)
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar([autoep_label, zero3_leaf_label], [a_tps, z_tps])
        ax.set_ylabel("Tokens/sec")
        ax.set_title("Average Throughput Comparison")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{bar.get_height():.0f}", ha="center", va="bottom")
        path = os.path.join(out_dir, "throughput_bar.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots["throughput_bar"] = path
    except Exception as exc:
        print(f"WARNING: throughput plot failed: {exc}")

    return plots


def sanitize(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {key: sanitize(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [sanitize(value) for value in obj]
    return obj


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    autoep_rows = [row for row in load_csv(args.autoep_csv) if int(row["step"]) >= args.warmup_steps]
    zero3_rows = [row for row in load_csv(args.zero3_leaf_csv) if int(row["step"]) >= args.warmup_steps]

    autoep_steps = {int(row["step"]): row for row in autoep_rows}
    zero3_steps = {int(row["step"]): row for row in zero3_rows}
    aligned_steps = sorted(set(autoep_steps) & set(zero3_steps))
    sufficient_evidence = len(aligned_steps) >= args.min_post_warmup_steps

    loss_parity = metric_parity(
        autoep_steps,
        zero3_steps,
        aligned_steps,
        "loss",
        args.min_post_warmup_steps,
    )

    a_avg_loss = sum(float(row["loss"]) for row in autoep_rows) / len(autoep_rows) if autoep_rows else 0
    z_avg_loss = sum(float(row["loss"]) for row in zero3_rows) / len(zero3_rows) if zero3_rows else 0
    a_peak_mem = max((int(row["cuda_peak_memory_allocated_bytes"]) for row in autoep_rows), default=0)
    z_peak_mem = max((int(row["cuda_peak_memory_allocated_bytes"]) for row in zero3_rows), default=0)
    a_avg_tps = sum(float(row["global_tokens_per_sec"]) for row in autoep_rows) / len(autoep_rows) if autoep_rows else 0
    z_avg_tps = sum(float(row["global_tokens_per_sec"]) for row in zero3_rows) / len(zero3_rows) if zero3_rows else 0

    threshold_passed = None
    if args.max_mean_abs_diff is not None and loss_parity["mean_abs_diff"] is not None:
        threshold_passed = loss_parity["mean_abs_diff"] <= args.max_mean_abs_diff

    plots = try_plot(
        autoep_rows,
        zero3_rows,
        args.out_dir,
        args.autoep_label,
        args.zero3_leaf_label,
    )

    summary = sanitize(
        {
            "aligned_steps": len(aligned_steps),
            "sufficient_evidence": sufficient_evidence,
            "loss": {
                "autoep_mean": a_avg_loss,
                "zero3_leaf_mean": z_avg_loss,
                "mean_abs_diff": loss_parity["mean_abs_diff"],
                "max_abs_diff": loss_parity["max_abs_diff"],
            },
            "loss_parity": loss_parity,
            "threshold_checks": {
                "max_mean_abs_diff": args.max_mean_abs_diff,
                "passed": threshold_passed,
            },
            "peak_memory": {
                "autoep_bytes": a_peak_mem,
                "zero3_leaf_bytes": z_peak_mem,
                "ratio": a_peak_mem / z_peak_mem if z_peak_mem > 0 else None,
            },
            "throughput": {
                "autoep_tokens_per_sec": a_avg_tps,
                "zero3_leaf_tokens_per_sec": z_avg_tps,
                "ratio": a_avg_tps / z_avg_tps if z_avg_tps > 0 else None,
            },
            "plots": plots,
        }
    )

    tmp = args.out_json + ".tmp"
    parent = os.path.dirname(args.out_json)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, args.out_json)

    print("\n=== Comparison Summary ===")
    print(f"Aligned steps: {len(aligned_steps)}")
    print(f"Mean loss: AutoEP={summary['loss']['autoep_mean']}, ZeRO-3={summary['loss']['zero3_leaf_mean']}")
    print(f"Mean abs diff (loss): {loss_parity['mean_abs_diff']}")
    print(f"Max abs diff (loss): {loss_parity['max_abs_diff']}")
    if loss_parity["pearson_correlation"] is not None:
        print(f"Pearson correlation (loss): {loss_parity['pearson_correlation']:.4f}")
    print(f"Peak memory ratio (AutoEP / ZeRO-3): {summary['peak_memory']['ratio']}")
    print(f"Throughput ratio (AutoEP / ZeRO-3): {summary['throughput']['ratio']}")
    print(f"Summary written to: {args.out_json}")


if __name__ == "__main__":
    main()
