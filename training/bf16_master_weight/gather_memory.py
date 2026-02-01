#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

"""
Script to gather and compare memory usage from training logs.

Usage:
    python gather_memory.py --log_dir logs/20231201_120000
"""

import argparse
import os
import re
from pathlib import Path


def parse_summary_line(line):
    """Parse the SUMMARY line from log output."""
    pattern = r"SUMMARY: config=(\S+) params=(\d+) peak_mem_bytes=(\d+) alloc_mem_bytes=(\d+) avg_step_time=(\S+)"
    match = re.search(pattern, line)
    if match:
        return {
            "config": match.group(1),
            "params": int(match.group(2)),
            "peak_mem_bytes": int(match.group(3)),
            "alloc_mem_bytes": int(match.group(4)),
            "avg_step_time": float(match.group(5)),
        }
    return None


def format_bytes(bytes_val):
    """Format bytes to human-readable string."""
    gb = bytes_val / (1024 ** 3)
    return f"{gb:.2f} GB"


def format_bytes_mb(bytes_val):
    """Format bytes to MB."""
    mb = bytes_val / (1024 ** 2)
    return f"{mb:.1f} MB"


def get_config_name(config_path):
    """Extract clean config name from path."""
    name = Path(config_path).stem
    if name == "baseline":
        return "Baseline (fp32 master)"
    elif name == "bf16_master_wg":
        return "bf16_master_weights_and_grads"
    elif name == "bf16_full":
        return "bf16_full (master + opt states)"
    return name


def main():
    parser = argparse.ArgumentParser(description="Gather memory usage from training logs")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory containing log files")
    parser.add_argument("--output", type=str, default=None, help="Output file for summary")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: Log directory '{log_dir}' does not exist")
        return 1

    # Find and parse all log files
    results = []
    log_files = ["baseline.log", "bf16_full.log"]

    for log_file in log_files:
        log_path = log_dir / log_file
        if not log_path.exists():
            print(f"Warning: Log file '{log_path}' not found, skipping")
            continue

        with open(log_path, "r") as f:
            for line in f:
                summary = parse_summary_line(line)
                if summary:
                    results.append(summary)
                    break

    if not results:
        print("No results found in log files")
        return 1

    # Calculate baseline for comparison
    baseline_peak = None
    for r in results:
        if "baseline" in r["config"]:
            baseline_peak = r["peak_mem_bytes"]
            break

    # Generate summary
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("BF16 Low-Precision Master Weights - Memory Usage Comparison")
    output_lines.append("=" * 80)
    output_lines.append("")

    # Table header
    output_lines.append(f"{'Configuration':<40} {'Peak Memory':<15} {'Reduction':<15} {'Step Time':<12}")
    output_lines.append("-" * 80)

    for r in results:
        config_name = get_config_name(r["config"])
        peak_mem = format_bytes(r["peak_mem_bytes"])
        step_time = f"{r['avg_step_time']:.4f}s"

        if baseline_peak and baseline_peak > 0:
            reduction = ((baseline_peak - r["peak_mem_bytes"]) / baseline_peak) * 100
            reduction_str = f"{reduction:+.1f}%" if reduction != 0 else "-"
        else:
            reduction_str = "-"

        output_lines.append(f"{config_name:<40} {peak_mem:<15} {reduction_str:<15} {step_time:<12}")

    output_lines.append("-" * 80)
    output_lines.append("")

    # Detailed breakdown
    output_lines.append("Detailed Results:")
    output_lines.append("-" * 40)
    for r in results:
        config_name = get_config_name(r["config"])
        output_lines.append(f"\n{config_name}:")
        output_lines.append(f"  Parameters: {r['params']:,}")
        output_lines.append(f"  Peak Memory: {format_bytes(r['peak_mem_bytes'])} ({r['peak_mem_bytes']:,} bytes)")
        output_lines.append(f"  Allocated Memory: {format_bytes(r['alloc_mem_bytes'])} ({r['alloc_mem_bytes']:,} bytes)")
        output_lines.append(f"  Avg Step Time: {r['avg_step_time']:.4f}s")

    output_lines.append("")
    output_lines.append("=" * 80)

    # Generate markdown table
    output_lines.append("")
    output_lines.append("Markdown Table (for README):")
    output_lines.append("-" * 40)
    output_lines.append("")
    output_lines.append("| Configuration | Peak Memory | Memory Reduction | Avg Step Time |")
    output_lines.append("|---------------|-------------|------------------|---------------|")

    for r in results:
        config_name = get_config_name(r["config"])
        peak_mem = format_bytes(r["peak_mem_bytes"])
        step_time = f"{r['avg_step_time']:.4f}s"

        if baseline_peak and baseline_peak > 0:
            reduction = ((baseline_peak - r["peak_mem_bytes"]) / baseline_peak) * 100
            reduction_str = f"{reduction:+.1f}%" if reduction != 0 else "-"
        else:
            reduction_str = "-"

        output_lines.append(f"| {config_name} | {peak_mem} | {reduction_str} | {step_time} |")

    output_lines.append("")

    # Print to stdout
    summary_text = "\n".join(output_lines)
    print(summary_text)

    # Save to file
    output_path = args.output or (log_dir / "summary.txt")
    with open(output_path, "w") as f:
        f.write(summary_text)

    print(f"\nSummary saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
