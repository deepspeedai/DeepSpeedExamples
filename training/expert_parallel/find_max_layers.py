"""Binary search for maximum stable layer count per mode.

Launches train.py as subprocesses with increasing layer counts to find
the maximum stable configuration for both AutoEP and ZeRO-3 leaf modes.

Run as a regular Python script (NOT via deepspeed launcher):
    python find_max_layers.py --autoep_size 8 --output_json runs/layer_search.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

from train import MODEL_PRESETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find max stable layers per mode")
    parser.add_argument("--min_layers", type=int, default=2)
    parser.add_argument("--max_layers", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--trial_steps", type=int, default=20)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--master_port", type=int, default=29600)
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3_5",
        choices=sorted(MODEL_PRESETS.keys()),
        help="Same as train.py --model (affects built-in AutoEP preset / leaf class)",
    )
    parser.add_argument(
        "--autoep_size",
        type=int,
        required=True,
        help="AutoEP size passed to train.py --autoep_size",
    )
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument(
        "--log_dir",
        type=str,
        default="runs/layer_search/",
    )
    parser.add_argument("--trial_timeout", type=int, default=300)
    parser.add_argument("--resume_from_json", type=str, default=None)
    return parser.parse_args()


def classify_failure(exit_code: int, log_content: str) -> tuple[str, str | None]:
    """Classify failure from exit code and log content."""
    if exit_code == 3:
        return "nan_inf", "Non-finite loss detected"
    if exit_code == 2:
        return "config", "Configuration or validation error"

    # Check log content for patterns
    oom_patterns = [
        r"CUDA out of memory",
        r"OutOfMemoryError",
        r"RuntimeError: CUDA error: out of memory",
    ]
    nccl_patterns = [
        r"NCCL error",
        r"NCCL.*timeout",
        r"ProcessGroupNCCL",
        r"ncclSystemError",
    ]

    for pattern in oom_patterns:
        match = re.search(pattern, log_content, re.IGNORECASE)
        if match:
            return "oom", match.group(0)[:200]

    for pattern in nccl_patterns:
        match = re.search(pattern, log_content, re.IGNORECASE)
        if match:
            return "nccl", match.group(0)[:200]

    if exit_code == -9 or exit_code == 137:
        return "oom", "Process killed (likely OOM)"

    return "other", f"Exit code {exit_code}"


def run_trial(
    mode: str,
    num_layers: int,
    args: argparse.Namespace,
    trial_idx: int,
    grad_accum: int,
    attempt: int = 1,
) -> dict:
    """Run a single trial and return the result dict."""
    master_port = args.master_port + trial_idx

    trial_id = f"{mode}_L{num_layers}_attempt{attempt}"
    log_dir = os.path.join(args.log_dir, trial_id)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "output.log")
    metrics_path = os.path.join(log_dir, f"metrics_{mode}.csv")

    cmd = [
        "deepspeed",
        "--num_gpus", str(args.num_gpus),
        "--master_port", str(master_port),
        "train.py",
        "--mode", mode,
        "--model", args.model,
        "--steps", str(args.trial_steps),
        "--warmup_steps", "2",
        "--num_layers", str(num_layers),
        "--seq_len", str(args.seq_len),
        "--micro_batch_size", str(args.micro_batch_size),
        "--grad_accum", str(grad_accum),
        "--seed", str(args.seed),
        "--metrics_out", metrics_path,
    ]
    if mode == "autoep":
        cmd.extend(["--autoep_size", str(args.autoep_size)])

    started_at = datetime.now(timezone.utc).isoformat()
    print(f"[{trial_id}] Starting: {num_layers} layers, port {master_port}")

    start_time = time.time()
    try:
        with open(log_path, "w") as log_file:
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=args.trial_timeout,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
        exit_code = result.returncode
    except subprocess.TimeoutExpired:
        exit_code = -1
        with open(log_path, "a") as f:
            f.write(f"\n[find_max_layers] TIMEOUT after {args.trial_timeout}s\n")
    duration = time.time() - start_time
    finished_at = datetime.now(timezone.utc).isoformat()

    # Read log for failure classification
    try:
        with open(log_path) as f:
            log_content = f.read()
    except Exception:
        log_content = ""

    if exit_code == 0:
        status = "success"
        failure_reason = None
    elif exit_code == -1:
        status = "timeout_hang"
        failure_reason = f"Timeout after {args.trial_timeout}s"
    else:
        status, failure_reason = classify_failure(exit_code, log_content)

    print(f"[{trial_id}] Result: {status} (exit_code={exit_code}, {duration:.1f}s)")

    return {
        "trial_id": trial_id,
        "attempt": attempt,
        "mode": mode,
        "num_layers": num_layers,
        "status": status,
        "failure_reason": failure_reason,
        "exit_code": exit_code,
        "master_port": master_port,
        "command": cmd,
        "log_path": log_path,
        "metrics_path": metrics_path if exit_code == 0 else None,
        "duration_sec": duration,
        "started_at": started_at,
        "finished_at": finished_at,
    }


def binary_search_max_layers(
    mode: str,
    args: argparse.Namespace,
    trial_counter: list[int],
    grad_accum: int,
    search_history: list[dict],
    known_results: dict[int, str] | None = None,
) -> int:
    """Binary search for max stable layer count."""
    if known_results is None:
        known_results = {}

    lo = args.min_layers
    hi = args.max_layers
    best = 0

    # Exponential probe phase: find upper bound
    probe = lo
    while probe <= hi:
        if probe in known_results:
            if known_results[probe] == "success":
                best = max(best, probe)
                probe *= 2
                continue
            else:
                hi = probe
                break

        trial_counter[0] += 1
        result = run_trial(mode, probe, args, trial_counter[0], grad_accum)
        search_history.append(result)
        known_results[probe] = result["status"]

        if result["status"] == "success":
            best = max(best, probe)
            probe *= 2
        else:
            # Retry once for transient NCCL failures
            if result["status"] == "nccl":
                trial_counter[0] += 1
                retry = run_trial(mode, probe, args, trial_counter[0], grad_accum, attempt=2)
                search_history.append(retry)
                if retry["status"] == "success":
                    known_results[probe] = "success"
                    best = max(best, probe)
                    probe *= 2
                    continue
            hi = probe
            break
    else:
        # All probes succeeded up to max_layers
        return min(best, args.max_layers) if best > 0 else 0

    if best == 0 and lo not in known_results:
        # Try minimum
        trial_counter[0] += 1
        result = run_trial(mode, lo, args, trial_counter[0], grad_accum)
        search_history.append(result)
        known_results[lo] = result["status"]
        if result["status"] == "success":
            best = lo

    if best == 0:
        return 0

    # Binary search between best and hi
    lo_bs = best
    hi_bs = hi
    while lo_bs < hi_bs - 1:
        mid = (lo_bs + hi_bs) // 2
        if mid in known_results:
            if known_results[mid] == "success":
                lo_bs = mid
                best = max(best, mid)
            else:
                hi_bs = mid
            continue

        trial_counter[0] += 1
        result = run_trial(mode, mid, args, trial_counter[0], grad_accum)
        search_history.append(result)
        known_results[mid] = result["status"]

        if result["status"] == "success":
            lo_bs = mid
            best = max(best, mid)
        else:
            hi_bs = mid

    return best


def write_output(data: dict, path: str) -> None:
    """Atomically write output JSON."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    # Load prior results if resuming
    prior_history = []
    if args.resume_from_json and os.path.exists(args.resume_from_json):
        with open(args.resume_from_json) as f:
            prior = json.load(f)
        prior_history = prior.get("search_history", [])
        print(f"Resuming from {len(prior_history)} prior trials")

    grad_accum_autoep = args.grad_accum
    grad_accum_zero3 = args.grad_accum

    search_history = list(prior_history)
    trial_counter = [len(prior_history)]

    output = {
        "max_layers_autoep": 0,
        "max_layers_zero3_leaf": 0,
        "final_layers": 0,
        "status": "running",
        "last_completed_trial_id": None,
        "search_history": search_history,
        "search_config": {
            "min_layers": args.min_layers,
            "max_layers": args.max_layers,
            "model": args.model,
            "seq_len": args.seq_len,
            "micro_batch_size": args.micro_batch_size,
            "autoep_size": args.autoep_size,
            "grad_accum_autoep": grad_accum_autoep,
            "grad_accum_zero3_leaf": grad_accum_zero3,
            "trial_steps": args.trial_steps,
            "trial_timeout": args.trial_timeout,
            "num_gpus": args.num_gpus,
            "base_master_port": args.master_port,
        },
    }
    write_output(output, args.output_json)

    # Search AutoEP
    print("\n=== Searching max layers for AutoEP ===")
    max_autoep = binary_search_max_layers(
        "autoep", args, trial_counter, grad_accum_autoep, search_history
    )
    output["max_layers_autoep"] = max_autoep
    print(f"AutoEP max layers: {max_autoep}")

    # Search ZeRO-3 leaf
    print("\n=== Searching max layers for ZeRO-3 leaf ===")
    max_zero3 = binary_search_max_layers(
        "zero3_leaf", args, trial_counter, grad_accum_zero3, search_history
    )
    output["max_layers_zero3_leaf"] = max_zero3
    print(f"ZeRO-3 leaf max layers: {max_zero3}")

    # Final
    final = min(max_autoep, max_zero3) if max_autoep > 0 and max_zero3 > 0 else 0
    output["final_layers"] = final
    output["status"] = "complete" if final > 0 else "failed"
    if search_history:
        output["last_completed_trial_id"] = search_history[-1]["trial_id"]

    write_output(output, args.output_json)

    print(f"\n=== Results ===")
    print(f"AutoEP max: {max_autoep}, ZeRO-3 leaf max: {max_zero3}, Final: {final}")
    print(f"Output: {args.output_json}")

    if final == 0:
        print("ERROR: No feasible layer count found for both modes.")
        sys.exit(1)


if __name__ == "__main__":
    main()
