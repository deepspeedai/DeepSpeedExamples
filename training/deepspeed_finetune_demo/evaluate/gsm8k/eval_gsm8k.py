#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate GSM8K accuracy from generated samples.jsonl.

Usage:
    python evaluate/gsm8k/eval_gsm8k.py --samples eval_results/gsm8k_baseline/samples.jsonl
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, required=True, help="Path to samples.jsonl")
    args = parser.parse_args()

    total = 0
    correct = 0
    empty = 0

    with open(args.samples) as f:
        for line in f:
            sample = json.loads(line)
            total += 1
            predicted = sample.get("predicted_answer", "")
            gold = sample["correct_answer"]
            if not predicted:
                empty += 1
            elif predicted == gold:
                correct += 1

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"GSM8K Results:")
    print(f"  Total:     {total}")
    print(f"  Correct:   {correct}")
    print(f"  Empty:     {empty}")
    print(f"  Accuracy:  {accuracy:.2f}%")
    print(f"  pass@1:    {accuracy:.2f}")


if __name__ == "__main__":
    main()
