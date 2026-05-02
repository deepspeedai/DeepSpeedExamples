"""Offline comparison of AutoEP vs ZeRO-3 leaf training metrics.

Reads CSV outputs from both modes and generates plots + summary JSON.

Run as a regular Python script (NOT via deepspeed launcher):
    python compare_metrics.py --autoep_csv metrics_autoep.csv --zero3_leaf_csv metrics_zero3_leaf.csv \
        --autoep_metadata run_metadata_autoep.json --zero3_leaf_metadata run_metadata_zero3_leaf.json \
        --out_dir results/ --out_json results/summary.json
"""

import argparse
import csv
import json
import math
import os
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare AutoEP and ZeRO-3 leaf training metrics"
    )
    parser.add_argument("--autoep_csv", type=str, required=True)
    parser.add_argument("--zero3_leaf_csv", type=str, required=True)
    parser.add_argument("--autoep_metadata", type=str, required=True)
    parser.add_argument("--zero3_leaf_metadata", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument(
        "--autoep_label", type=str, default="AutoEP + ZeRO-1"
    )
    parser.add_argument(
        "--zero3_leaf_label", type=str, default="HF + ZeRO-3 leaf"
    )
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--max_mean_abs_diff", type=float, default=None)
    parser.add_argument("--min_post_warmup_steps", type=int, default=10)
    hash_group = parser.add_mutually_exclusive_group()
    hash_group.add_argument(
        "--require_same_init_hash",
        dest="require_same_init_hash",
        action="store_true",
        help="Require matching non-empty init_weights_sha256 in both metadata files",
    )
    hash_group.add_argument(
        "--no_require_same_init_hash",
        dest="require_same_init_hash",
        action="store_false",
        help="Allow comparison without enforcing matching init_weights_sha256",
    )
    parser.set_defaults(require_same_init_hash=True)
    return parser.parse_args()


def load_csv(path: str) -> list[dict]:
    """Load CSV and return list of row dicts."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_metadata(path: str) -> dict:
    """Load metadata JSON."""
    with open(path) as f:
        return json.load(f)


def parse_optional_float(value: Any) -> float | None:
    """Parse a metric value, treating missing/blank values as unavailable."""
    if value is None or value == "":
        return None
    return float(value)


def pearson_correlation(x: list[float], y: list[float]) -> float | None:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return None
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    dx = [xi - mean_x for xi in x]
    dy = [yi - mean_y for yi in y]
    num = sum(a * b for a, b in zip(dx, dy))
    den_x = math.sqrt(sum(a * a for a in dx))
    den_y = math.sqrt(sum(b * b for b in dy))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def metric_parity(
    autoep_steps: dict[int, dict],
    zero3_steps: dict[int, dict],
    aligned_steps: list[int],
    metric_name: str,
    min_corr_steps: int,
    optional: bool = False,
) -> dict[str, Any]:
    """Compute aligned parity stats for one numeric metric column."""
    metric_steps = []
    a_values = []
    z_values = []
    for step in aligned_steps:
        if optional:
            a_value = parse_optional_float(autoep_steps[step].get(metric_name))
            z_value = parse_optional_float(zero3_steps[step].get(metric_name))
            if a_value is None or z_value is None:
                continue
        else:
            a_value = float(autoep_steps[step][metric_name])
            z_value = float(zero3_steps[step][metric_name])
        metric_steps.append(step)
        a_values.append(a_value)
        z_values.append(z_value)

    if not metric_steps:
        return {
            "recorded": False,
            "note": f"{metric_name} was not recorded in both CSVs for aligned steps.",
            "mean_abs_diff": float("nan"),
            "max_abs_diff": float("nan"),
            "pearson_correlation": None,
            "num_aligned_steps": 0,
        }

    abs_diffs = [abs(a - z) for a, z in zip(a_values, z_values)]
    return {
        "recorded": True,
        "note": None,
        "mean_abs_diff": sum(abs_diffs) / len(abs_diffs),
        "max_abs_diff": max(abs_diffs),
        "pearson_correlation": (
            pearson_correlation(a_values, z_values)
            if len(metric_steps) >= min_corr_steps
            else None
        ),
        "num_aligned_steps": len(metric_steps),
    }


def validate_compatibility(
    autoep_meta: dict, zero3_meta: dict, require_same_init_hash: bool
) -> tuple[bool, list[str], list[str], bool | None, str | None, str | None]:
    """Check if runs are comparable."""
    issues = []
    warnings = []

    # Check num_layers
    a_layers = autoep_meta.get("args", {}).get("num_layers")
    z_layers = zero3_meta.get("args", {}).get("num_layers")
    if a_layers != z_layers:
        issues.append(f"num_layers mismatch: autoep={a_layers}, zero3={z_layers}")

    # Check seq_len
    a_seq = autoep_meta.get("args", {}).get("seq_len")
    z_seq = zero3_meta.get("args", {}).get("seq_len")
    if a_seq != z_seq:
        issues.append(f"seq_len mismatch: autoep={a_seq}, zero3={z_seq}")

    # Check effective tokens per update
    a_tokens = autoep_meta.get("effective_tokens_per_update")
    z_tokens = zero3_meta.get("effective_tokens_per_update")
    if a_tokens != z_tokens:
        issues.append(
            f"effective_tokens_per_update mismatch: autoep={a_tokens}, zero3={z_tokens}"
        )

    # Check precision
    a_args = autoep_meta.get("args", {})
    z_args = zero3_meta.get("args", {})

    # Check world_size
    a_ws = autoep_meta.get("world_size")
    z_ws = zero3_meta.get("world_size")
    if a_ws != z_ws:
        issues.append(f"world_size mismatch: autoep={a_ws}, zero3={z_ws}")

    a_init_hash = autoep_meta.get("init_weights_sha256")
    z_init_hash = zero3_meta.get("init_weights_sha256")
    same_init_hash = (
        a_init_hash == z_init_hash if a_init_hash and z_init_hash else None
    )

    if require_same_init_hash:
        if not a_init_hash or not z_init_hash:
            issues.append(
                "init_weights_sha256 missing in one or both metadata files "
                "(required by --require_same_init_hash)."
            )
        elif a_init_hash != z_init_hash:
            issues.append(
                f"init_weights_sha256 mismatch: autoep={a_init_hash}, zero3={z_init_hash}"
            )
    else:
        if not a_init_hash or not z_init_hash:
            warnings.append(
                "init_weights_sha256 missing in one or both metadata files; "
                "init-weight provenance is not verified."
            )
        elif a_init_hash != z_init_hash:
            warnings.append(
                "init_weights_sha256 mismatch detected but allowed by "
                "--no_require_same_init_hash."
            )

    return (
        len(issues) == 0,
        issues,
        warnings,
        same_init_hash,
        a_init_hash,
        z_init_hash,
    )


def try_plot(
    autoep_rows: list[dict],
    zero3_rows: list[dict],
    out_dir: str,
    autoep_label: str,
    zero3_leaf_label: str,
) -> dict[str, str | None]:
    """Generate comparison plots. Returns dict of plot paths or None."""
    plots = {
        "loss_curve": None,
        "ce_loss_curve": None,
        "total_loss_curve": None,
        "aux_loss_curve": None,
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

    def plot_metric_curve(
        metric_name: str,
        ylabel: str,
        title: str,
        filename: str,
        optional: bool = False,
    ) -> str | None:
        a_points = []
        z_points = []
        for row in autoep_rows:
            value = (
                parse_optional_float(row.get(metric_name))
                if optional
                else float(row[metric_name])
            )
            if value is not None:
                a_points.append((int(row["step"]), value))
        for row in zero3_rows:
            value = (
                parse_optional_float(row.get(metric_name))
                if optional
                else float(row[metric_name])
            )
            if value is not None:
                z_points.append((int(row["step"]), value))
        if not a_points or not z_points:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            [p[0] for p in a_points],
            [p[1] for p in a_points],
            label=autoep_label,
            marker="o",
            markersize=3,
        )
        ax.plot(
            [p[0] for p in z_points],
            [p[1] for p in z_points],
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

    # CE loss curve. Keep the legacy loss_curve key/path for existing docs.
    try:
        path = plot_metric_curve(
            "loss_ce",
            "CE Loss",
            "CE Loss Curve Comparison",
            "loss_curve.png",
        )
        plots["loss_curve"] = path
        plots["ce_loss_curve"] = path
    except Exception as e:
        print(f"WARNING: CE loss curve plot failed: {e}")

    # Total loss curve
    try:
        plots["total_loss_curve"] = plot_metric_curve(
            "loss_total",
            "Total Loss",
            "Total Loss Curve Comparison",
            "total_loss_curve.png",
        )
    except Exception as e:
        print(f"WARNING: Total loss curve plot failed: {e}")

    # Aux-loss curve when both CSVs recorded it
    try:
        plots["aux_loss_curve"] = plot_metric_curve(
            "loss_aux",
            "Aux Loss",
            "Router Aux-Loss Curve Comparison",
            "aux_loss_curve.png",
            optional=True,
        )
    except Exception as e:
        print(f"WARNING: Aux-loss curve plot failed: {e}")

    # Peak memory bar
    try:
        a_peak = max(int(r["cuda_peak_memory_allocated_bytes"]) for r in autoep_rows)
        z_peak = max(int(r["cuda_peak_memory_allocated_bytes"]) for r in zero3_rows)

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            [autoep_label, zero3_leaf_label],
            [a_peak / 1e9, z_peak / 1e9],
            color=["#2196F3", "#FF9800"],
        )
        ax.set_ylabel("Peak Memory (GB)")
        ax.set_title("Peak GPU Memory Comparison")
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )
        path = os.path.join(out_dir, "peak_memory_bar.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots["peak_memory_bar"] = path
    except Exception as e:
        print(f"WARNING: Peak memory plot failed: {e}")

    # Throughput bar
    try:
        a_tps = [float(r["global_tokens_per_sec"]) for r in autoep_rows]
        z_tps = [float(r["global_tokens_per_sec"]) for r in zero3_rows]
        a_avg = sum(a_tps) / len(a_tps) if a_tps else 0
        z_avg = sum(z_tps) / len(z_tps) if z_tps else 0

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            [autoep_label, zero3_leaf_label],
            [a_avg, z_avg],
            color=["#2196F3", "#FF9800"],
        )
        ax.set_ylabel("Tokens/sec")
        ax.set_title("Average Throughput Comparison (post-warmup)")
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.0f}",
                ha="center",
                va="bottom",
            )
        path = os.path.join(out_dir, "throughput_bar.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots["throughput_bar"] = path
    except Exception as e:
        print(f"WARNING: Throughput plot failed: {e}")

    return plots


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    autoep_rows = load_csv(args.autoep_csv)
    zero3_rows = load_csv(args.zero3_leaf_csv)
    autoep_meta = load_metadata(args.autoep_metadata)
    zero3_meta = load_metadata(args.zero3_leaf_metadata)

    # Validate compatibility
    (
        compatible,
        compat_issues,
        compat_warnings,
        same_init_hash,
        init_hash_autoep,
        init_hash_zero3_leaf,
    ) = validate_compatibility(
        autoep_meta,
        zero3_meta,
        require_same_init_hash=args.require_same_init_hash,
    )

    # Filter by warmup steps
    autoep_rows = [r for r in autoep_rows if int(r["step"]) >= args.warmup_steps]
    zero3_rows = [r for r in zero3_rows if int(r["step"]) >= args.warmup_steps]

    # Align steps
    autoep_steps = {int(r["step"]): r for r in autoep_rows}
    zero3_steps = {int(r["step"]): r for r in zero3_rows}
    aligned_steps = sorted(set(autoep_steps.keys()) & set(zero3_steps.keys()))

    num_aligned = len(aligned_steps)
    sufficient_evidence = num_aligned >= args.min_post_warmup_steps

    total_loss_parity = metric_parity(
        autoep_steps,
        zero3_steps,
        aligned_steps,
        "loss_total",
        args.min_post_warmup_steps,
    )
    ce_loss_parity = metric_parity(
        autoep_steps,
        zero3_steps,
        aligned_steps,
        "loss_ce",
        args.min_post_warmup_steps,
    )
    aux_loss_parity = metric_parity(
        autoep_steps,
        zero3_steps,
        aligned_steps,
        "loss_aux",
        3,
        optional=True,
    )

    # Check loss objective tag compatibility
    objective_mismatch = False
    if aligned_steps:
        a_tags = {autoep_steps[s].get("loss_objective_tag", "") for s in aligned_steps}
        z_tags = {zero3_steps[s].get("loss_objective_tag", "") for s in aligned_steps}
        if a_tags != z_tags:
            objective_mismatch = True

    # Threshold checks
    threshold_passed = None
    threshold_skipped_reason = None
    if args.max_mean_abs_diff is not None:
        if not compatible:
            threshold_passed = None
            threshold_skipped_reason = "Runs not comparable: " + "; ".join(compat_issues)
        elif objective_mismatch:
            threshold_passed = None
            threshold_skipped_reason = (
                "Loss objective tag mismatch between modes; threshold check skipped."
            )
        elif not sufficient_evidence:
            threshold_passed = None
            threshold_skipped_reason = (
                f"Insufficient aligned post-warmup steps ({num_aligned} < {args.min_post_warmup_steps})"
            )
        else:
            threshold_passed = (
                ce_loss_parity["mean_abs_diff"] <= args.max_mean_abs_diff
            )

    # Peak memory
    a_peak_mem = (
        max(int(r["cuda_peak_memory_allocated_bytes"]) for r in autoep_rows)
        if autoep_rows
        else 0
    )
    z_peak_mem = (
        max(int(r["cuda_peak_memory_allocated_bytes"]) for r in zero3_rows)
        if zero3_rows
        else 0
    )
    mem_ratio = a_peak_mem / z_peak_mem if z_peak_mem > 0 else float("nan")

    # Throughput
    a_tps_vals = [float(r["global_tokens_per_sec"]) for r in autoep_rows]
    z_tps_vals = [float(r["global_tokens_per_sec"]) for r in zero3_rows]
    a_avg_tps = sum(a_tps_vals) / len(a_tps_vals) if a_tps_vals else 0
    z_avg_tps = sum(z_tps_vals) / len(z_tps_vals) if z_tps_vals else 0
    tps_ratio = a_avg_tps / z_avg_tps if z_avg_tps > 0 else float("nan")

    # Generate plots
    plots = try_plot(
        autoep_rows,
        zero3_rows,
        args.out_dir,
        autoep_label=args.autoep_label,
        zero3_leaf_label=args.zero3_leaf_label,
    )

    # Build summary
    summary = {
        "compatible": compatible,
        "compatibility_issues": compat_issues,
        "compatibility_warnings": compat_warnings,
        "same_init_hash": same_init_hash,
        "init_hash_autoep": init_hash_autoep,
        "init_hash_zero3_leaf": init_hash_zero3_leaf,
        "init_hash_required": args.require_same_init_hash,
        "loss_parity": {
            **ce_loss_parity,
            "metric": "loss_ce",
            "num_post_warmup_steps": num_aligned,
            "sufficient_evidence": sufficient_evidence,
        },
        "total_loss_parity": {
            **total_loss_parity,
            "metric": "loss_total",
            "num_post_warmup_steps": num_aligned,
            "sufficient_evidence": sufficient_evidence,
        },
        "ce_loss_parity": {
            **ce_loss_parity,
            "metric": "loss_ce",
            "num_post_warmup_steps": num_aligned,
            "sufficient_evidence": sufficient_evidence,
        },
        "aux_loss_parity": {
            **aux_loss_parity,
            "metric": "loss_aux",
        },
        "threshold_checks": {
            "max_mean_abs_diff": args.max_mean_abs_diff,
            "passed": threshold_passed,
            "skipped_reason": threshold_skipped_reason,
        },
        "peak_memory": {
            "autoep_bytes": a_peak_mem,
            "zero3_leaf_bytes": z_peak_mem,
            "ratio": mem_ratio,
        },
        "throughput": {
            "autoep_tokens_per_sec": a_avg_tps,
            "zero3_leaf_tokens_per_sec": z_avg_tps,
            "ratio": tps_ratio,
        },
        "caveats": [
            "Throughput and memory comparisons include differing ZeRO stages "
            "and are not an isolated AutoEP-only benchmark.",
            "Loss comparison uses trend agreement, not bit-identical values. "
            "Small divergence is expected from different ZeRO stages and FP reduction order.",
        ],
        "autoep_metadata": autoep_meta,
        "zero3_leaf_metadata": zero3_meta,
        "plots": plots,
    }

    # Handle NaN for JSON serialization
    def sanitize(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    summary = sanitize(summary)

    # Write summary JSON atomically
    tmp = args.out_json + ".tmp"
    parent = os.path.dirname(args.out_json)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, args.out_json)

    # Print summary
    print("\n=== Comparison Summary ===")
    print(f"Compatible: {compatible}")
    if compat_issues:
        print(f"Issues: {compat_issues}")
    if compat_warnings:
        print(f"Warnings: {compat_warnings}")
    print(f"Init hash required: {args.require_same_init_hash}")
    print(f"Same init hash: {same_init_hash}")
    print(f"Aligned steps: {num_aligned}")
    print(f"Mean abs diff (total loss): {total_loss_parity['mean_abs_diff']}")
    print(f"Max abs diff (total loss): {total_loss_parity['max_abs_diff']}")
    if total_loss_parity["pearson_correlation"] is not None:
        print(
            "Pearson correlation (total loss): "
            f"{total_loss_parity['pearson_correlation']:.4f}"
        )
    print(f"Mean abs diff (CE loss): {ce_loss_parity['mean_abs_diff']}")
    print(f"Max abs diff (CE loss): {ce_loss_parity['max_abs_diff']}")
    if ce_loss_parity["pearson_correlation"] is not None:
        print(
            "Pearson correlation (CE loss): "
            f"{ce_loss_parity['pearson_correlation']:.4f}"
        )
    print(f"Aux loss recorded on aligned steps: {aux_loss_parity['num_aligned_steps']}")
    if aux_loss_parity["recorded"]:
        print(f"Mean abs diff (aux loss): {aux_loss_parity['mean_abs_diff']}")
        print(f"Max abs diff (aux loss): {aux_loss_parity['max_abs_diff']}")
        if aux_loss_parity["pearson_correlation"] is not None:
            print(
                "Pearson correlation (aux loss): "
                f"{aux_loss_parity['pearson_correlation']:.4f}"
            )
    print(f"Peak memory ratio (autoep/zero3): {mem_ratio}")
    print(f"Throughput ratio (autoep/zero3): {tps_ratio}")
    if threshold_passed is not None:
        print(f"Threshold check: {'PASSED' if threshold_passed else 'FAILED'}")
    print(f"\nSummary written to: {args.out_json}")

    if plots["loss_curve"]:
        print(f"CE loss curve plot: {plots['loss_curve']}")
    if plots["total_loss_curve"]:
        print(f"Total loss curve plot: {plots['total_loss_curve']}")
    if plots["aux_loss_curve"]:
        print(f"Aux loss curve plot: {plots['aux_loss_curve']}")
    if plots["peak_memory_bar"]:
        print(f"Memory plot: {plots['peak_memory_bar']}")
    if plots["throughput_bar"]:
        print(f"Throughput plot: {plots['throughput_bar']}")

    # Exit with non-zero for compatibility failures or threshold failure.
    if not compatible or threshold_passed is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
