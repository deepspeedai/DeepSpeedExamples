# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Diff the loss curves of two OPSD runs that differ only in AutoTP size.

Tensor parallelism partitions a layer's arithmetic across ranks; it is not
supposed to change what the layer computes. So a run at AutoTP=3 and a run at
AutoTP=1, given the same seed, data order and hyperparameters, must follow the
same loss curve to within floating-point reassociation error. A shard that is
cut on the wrong boundary still trains -- it just trains a different model --
so a smoke test that only checks for a clean exit will not catch it. This does.

The tolerance is relative, and the check is per-step rather than on the final
loss alone: reassociation error should jitter around zero, while a genuine
sharding bug compounds as the two runs' weights drift apart.

The first step is checked separately and much more tightly: both runs start from
the same checkpoint and no optimizer step has happened yet, so a gap there is a
wrong forward rather than accumulated drift, and training dynamics cannot be
blamed for it.

Over a long run the two curves also drift apart under their own dynamics: the
loss surface is not flat, so a 1e-7 gap at step 0 does not stay 1e-7 forever
even when both runs are correct. The check therefore reports the whole
trajectory and fails only on a gap that a correct implementation cannot produce.

Usage:
    python compare_loss.py runs/autotp1/metrics.jsonl runs/autotp3/metrics.jsonl
"""

import argparse
import json
import sys


def load(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", help="metrics.jsonl of the AutoTP=1 run")
    parser.add_argument("candidate", help="metrics.jsonl of the sharded run")
    parser.add_argument("--rtol", type=float, default=1e-2,
                        help="largest tolerated relative loss gap at any step")
    parser.add_argument("--forward-rtol", type=float, default=1e-6,
                        help="largest tolerated relative gap at the first step, before any "
                             "optimizer step has had a chance to amplify it")
    parser.add_argument("--print-every", type=int, default=50,
                        help="print one row every N steps (the worst step is always printed)")
    args = parser.parse_args()

    baseline = {r["step"]: r for r in load(args.baseline)}
    candidate = {r["step"]: r for r in load(args.candidate)}

    shared = sorted(set(baseline) & set(candidate))
    if not shared:
        print("no steps in common between the two runs", file=sys.stderr)
        return 1
    if len(shared) != len(baseline) or len(shared) != len(candidate):
        print(f"warning: comparing {len(shared)} shared steps "
              f"({len(baseline)} baseline, {len(candidate)} candidate)", file=sys.stderr)

    gaps = []
    for step in shared:
        b = baseline[step]["loss"]
        c = candidate[step]["loss"]
        gaps.append((step, b, c, abs(c - b) / max(abs(b), 1e-12)))

    worst_step, _, _, worst = max(gaps, key=lambda row: row[3])

    print(f"{'step':>6} {'baseline':>13} {'candidate':>13} {'rel':>11}")
    for step, b, c, rel in gaps:
        if step % args.print_every == 0 or step == shared[-1] or step == worst_step:
            marker = "  <- worst" if step == worst_step else ""
            print(f"{step:>6} {b:>13.6f} {c:>13.6f} {rel:>11.2e}{marker}")

    mean = sum(row[3] for row in gaps) / len(gaps)
    first_step, _, _, first_rel = gaps[0]
    print(f"\nsteps={len(gaps)} first_rel={first_rel:.2e} mean_rel={mean:.2e} "
          f"worst_rel={worst:.2e} at step {worst_step}")

    # The first step is the sharpest signal available: both runs start from the same
    # checkpoint and no optimizer step has happened yet, so training dynamics cannot
    # have amplified anything. A gap here is a wrong forward, not accumulated drift.
    if first_rel > args.forward_rtol:
        print(f"FAIL: the runs already disagree by {first_rel:.2e} at step {first_step}, "
              f"before any weight update -- the sharded forward is wrong "
              f"(tolerance {args.forward_rtol:.0e})",
              file=sys.stderr)
        return 1

    if worst > args.rtol:
        print(f"FAIL: loss diverges by {worst:.2e} at step {worst_step} "
              f"(tolerance {args.rtol:.0e})", file=sys.stderr)
        return 1
    print(f"OK: {len(gaps)} steps agree within {args.rtol:.0e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
