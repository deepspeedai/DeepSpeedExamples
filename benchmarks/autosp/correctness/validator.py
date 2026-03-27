"""
Validates that per-rank losses from autosp match the baseline (compiled DS-Ulysses)
within a configurable threshold.

Reads two JSON files produced by correctness_run.py and compares them element-wise.
"""

import argparse
import json
import sys


def validate(baseline_path, autosp_path, threshold):
    with open(baseline_path) as f:
        baseline_data = json.load(f)
    with open(autosp_path) as f:
        autosp_data = json.load(f)

    baseline_losses = baseline_data["losses"]
    autosp_losses = autosp_data["losses"]

    baseline_ranks = sorted(baseline_losses.keys(), key=int)
    autosp_ranks = sorted(autosp_losses.keys(), key=int)

    if baseline_ranks != autosp_ranks:
        print(
            f"  FAIL: Rank mismatch — "
            f"baseline has ranks {baseline_ranks}, autosp has ranks {autosp_ranks}"
        )
        return False

    all_pass = True
    max_diff = 0.0
    mismatches = []

    for rank in baseline_ranks:
        bl_steps = baseline_losses[rank]
        asp_steps = autosp_losses[rank]

        all_steps = sorted(set(bl_steps.keys()) | set(asp_steps.keys()), key=int)
        for step in all_steps:
            if step not in bl_steps:
                mismatches.append(f"    Rank {rank}, Step {step}: missing in baseline")
                all_pass = False
                continue
            if step not in asp_steps:
                mismatches.append(f"    Rank {rank}, Step {step}: missing in autosp")
                all_pass = False
                continue

            bl_val = bl_steps[step]
            asp_val = asp_steps[step]
            diff = abs(bl_val - asp_val)
            max_diff = max(max_diff, diff)

            if diff > threshold:
                mismatches.append(
                    f"    Rank {rank}, Step {step}: "
                    f"baseline={bl_val:.6f}, autosp={asp_val:.6f}, diff={diff:.6e}"
                )
                all_pass = False

    if all_pass:
        print(f"  PASS (max diff: {max_diff:.6e}, threshold: {threshold:.6e})")
    else:
        print(f"  FAIL (max diff: {max_diff:.6e}, threshold: {threshold:.6e})")
        for m in mismatches:
            print(m)

    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="Validate autosp losses against baseline"
    )
    parser.add_argument(
        "--baseline", required=True, help="Path to baseline losses JSON"
    )
    parser.add_argument("--autosp", required=True, help="Path to autosp losses JSON")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-2,
        help="Maximum allowed absolute difference per loss value (default: 1e-2)",
    )
    args = parser.parse_args()

    passed = validate(args.baseline, args.autosp, args.threshold)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
