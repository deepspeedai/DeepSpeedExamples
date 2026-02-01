#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

"""
Plot loss curves from training logs.

Usage:
    python plot_loss.py --baseline logs/baseline_loss.csv --bf16 logs/bf16_loss.csv --output loss_comparison.png
"""

import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np


def load_loss_data(filepath):
    """Load loss data from CSV file."""
    steps = []
    losses = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            losses.append(float(row['loss']))
    return np.array(steps), np.array(losses)


def smooth_curve(values, window=10):
    """Apply moving average smoothing."""
    if len(values) < window:
        return values
    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
    # Pad the beginning to maintain length
    padding = np.full(window-1, smoothed[0])
    return np.concatenate([padding, smoothed])


def main():
    parser = argparse.ArgumentParser(description="Plot loss curves comparison")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline loss CSV file")
    parser.add_argument("--bf16", type=str, required=True, help="BF16 low-precision loss CSV file")
    parser.add_argument("--output", type=str, default="loss_comparison.png", help="Output plot file")
    parser.add_argument("--smooth", type=int, default=20, help="Smoothing window size")
    parser.add_argument("--title", type=str, default="Loss Comparison: Baseline vs BF16 Low-Precision", help="Plot title")
    args = parser.parse_args()

    # Load data
    baseline_steps, baseline_losses = load_loss_data(args.baseline)
    bf16_steps, bf16_losses = load_loss_data(args.bf16)

    # Apply smoothing
    baseline_smooth = smooth_curve(baseline_losses, args.smooth)
    bf16_smooth = smooth_curve(bf16_losses, args.smooth)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot raw data with low alpha
    ax.plot(baseline_steps, baseline_losses, alpha=0.2, color='blue')
    ax.plot(bf16_steps, bf16_losses, alpha=0.2, color='orange')

    # Plot smoothed curves
    ax.plot(baseline_steps, baseline_smooth, label='Baseline (fp32 master)', color='blue', linewidth=2)
    ax.plot(bf16_steps, bf16_smooth, label='BF16 Low-Precision', color='orange', linewidth=2)

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(args.title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add final loss values as text
    final_baseline = baseline_losses[-1]
    final_bf16 = bf16_losses[-1]
    textstr = f'Final Loss:\n  Baseline: {final_baseline:.4f}\n  BF16: {final_bf16:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved to: {args.output}")

    # Print statistics
    print(f"\nStatistics:")
    print(f"  Baseline - Final loss: {final_baseline:.4f}, Mean: {baseline_losses.mean():.4f}, Std: {baseline_losses.std():.4f}")
    print(f"  BF16     - Final loss: {final_bf16:.4f}, Mean: {bf16_losses.mean():.4f}, Std: {bf16_losses.std():.4f}")


if __name__ == "__main__":
    main()
