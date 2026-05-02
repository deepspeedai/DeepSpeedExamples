"""Run and summarize the issue #80 observed-results benchmark matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


MODELS: list[tuple[str, str]] = [
    ("qwen3_5", "Qwen3.5 MoE"),
    ("llama4", "Llama4"),
    ("mixtral_8x7b", "Mixtral"),
]
MODES = ["zero3_leaf", "autoep"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/mnt/local_storage/qwen35_kernel_speed_aux_20260502/task_e_observed_results_models",
    )
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--base_port", type=int, default=29180)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", default="wikitext")
    parser.add_argument("--dataset_percentage", type=float, default=10.0)
    parser.add_argument("--timeout_sec", type=int, default=3600)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str], log_path: Path, timeout_sec: int) -> tuple[int, float, str]:
    start = time.time()
    status = "complete"
    with log_path.open("w") as log_file:
        log_file.write(f"===== run started {utc_now()} =====\n")
        log_file.write("command=" + " ".join(cmd) + "\n\n")
        log_file.flush()
        try:
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
                check=False,
            )
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            returncode = -1
            status = f"timeout after {timeout_sec}s"
            log_file.write(f"\n[run_observed_results] TIMEOUT after {timeout_sec}s\n")
        finally:
            duration = time.time() - start
            log_file.write(
                f"\n===== run finished {utc_now()} returncode={returncode} "
                f"duration_sec={duration:.3f} status={status} =====\n"
            )
    return returncode, duration, status


def load_rows(metrics_csv: Path, warmup_steps: int) -> list[dict[str, str]]:
    with metrics_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if int(row["step"]) >= warmup_steps]


def summarize_success(
    *,
    model: str,
    model_label: str,
    mode: str,
    run_dir: Path,
    metrics_csv: Path,
    metadata_json: Path,
    warmup_steps: int,
    returncode: int,
    duration_sec: float,
    command: list[str],
    log_path: Path,
) -> dict[str, Any]:
    rows = load_rows(metrics_csv, warmup_steps)
    if not rows:
        raise ValueError(f"No post-warmup rows in {metrics_csv}")
    with metadata_json.open() as f:
        metadata = json.load(f)
    global_tps = [float(row["global_tokens_per_sec"]) for row in rows]
    peak_memory = [
        float(row["cuda_peak_memory_allocated_bytes"]) / (1024**3) for row in rows
    ]
    return {
        "status": "complete",
        "model": model,
        "model_label": model_label,
        "mode": mode,
        "returncode": returncode,
        "duration_sec": duration_sec,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "metrics_csv": str(metrics_csv),
        "metadata_json": str(metadata_json),
        "command": command,
        "first_metric_step": int(rows[0]["step"]),
        "last_metric_step": int(rows[-1]["step"]),
        "metric_rows": len(rows),
        "mean_global_tokens_per_sec": mean(global_tps),
        "peak_cuda_memory_allocated_gb": max(peak_memory),
        "world_size": metadata.get("world_size"),
        "dp_world_size": metadata.get("dp_world_size"),
        "autoep_size": metadata.get("autoep_size"),
        "effective_tokens_per_update": metadata.get("effective_tokens_per_update"),
        "loss_objective_tag": rows[-1].get("loss_objective_tag"),
        "loss_aux_recorded": any(row.get("loss_aux") not in (None, "") for row in rows),
    }


def summarize_failure(
    *,
    model: str,
    model_label: str,
    mode: str,
    run_dir: Path,
    returncode: int,
    duration_sec: float,
    status: str,
    command: list[str],
    log_path: Path,
) -> dict[str, Any]:
    tail = ""
    error = status
    if log_path.exists():
        text = log_path.read_text(errors="replace")
        tail = text[-4000:]
        for line in text.splitlines():
            if any(
                marker in line
                for marker in (
                    "ERROR: Unhandled exception:",
                    "CUDA out of memory",
                    "OutOfMemoryError",
                    "Traceback",
                    "exits with return code",
                )
            ):
                error = line
                break
    if returncode not in (0, -1) and error == "complete":
        error = f"subprocess failed with returncode {returncode}"
    return {
        "status": "failed" if returncode != -1 else "timeout",
        "model": model,
        "model_label": model_label,
        "mode": mode,
        "returncode": returncode,
        "duration_sec": duration_sec,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "command": command,
        "error": error,
        "log_tail": tail,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "model_label",
        "mode",
        "status",
        "mean_global_tokens_per_sec",
        "peak_cuda_memory_allocated_gb",
        "world_size",
        "dp_world_size",
        "autoep_size",
        "effective_tokens_per_update",
        "metric_rows",
        "returncode",
        "run_dir",
        "log_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    env_summary = {
        "created_at_utc": utc_now(),
        "root": str(root),
        "models": [model for model, _ in MODELS],
        "modes": MODES,
        "parameters": {
            "num_layers": args.num_layers,
            "micro_batch_size": args.micro_batch_size,
            "seq_len": args.seq_len,
            "grad_accum": args.grad_accum,
            "output_router_logits": True,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "gradient_checkpointing": "off",
            "shared_init": None,
        },
    }
    write_json(root / "manifest.json", env_summary)

    run_index = 0
    for model, model_label in MODELS:
        for mode in MODES:
            port = args.base_port + run_index
            run_dir = root / "runs" / model / mode
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_csv = run_dir / "metrics.csv"
            metadata_json = run_dir / "metadata.json"
            log_path = run_dir / "train.log"
            cmd = [
                "deepspeed",
                "--num_gpus",
                str(args.num_gpus),
                "--master_port",
                str(port),
                "train.py",
                "--model",
                model,
                "--mode",
                mode,
                "--num_layers",
                str(args.num_layers),
                "--steps",
                str(args.steps),
                "--warmup_steps",
                str(args.warmup_steps),
                "--log_interval",
                "1",
                "--seq_len",
                str(args.seq_len),
                "--micro_batch_size",
                str(args.micro_batch_size),
                "--grad_accum",
                str(args.grad_accum),
                "--seed",
                str(args.seed),
                "--dataset_name",
                args.dataset_name,
                "--dataset_percentage",
                str(args.dataset_percentage),
                "--gradient_checkpointing",
                "off",
                "--output_router_logits",
                "true",
                "--metrics_out",
                str(metrics_csv),
                "--run_metadata_out",
                str(metadata_json),
            ]
            if args.dry_run:
                result = {
                    "status": "dry_run",
                    "model": model,
                    "model_label": model_label,
                    "mode": mode,
                    "command": cmd,
                    "run_dir": str(run_dir),
                    "log_path": str(log_path),
                }
            else:
                returncode, duration_sec, status = run_command(
                    cmd, log_path, args.timeout_sec
                )
                if returncode == 0 and metrics_csv.exists() and metadata_json.exists():
                    try:
                        result = summarize_success(
                            model=model,
                            model_label=model_label,
                            mode=mode,
                            run_dir=run_dir,
                            metrics_csv=metrics_csv,
                            metadata_json=metadata_json,
                            warmup_steps=args.warmup_steps,
                            returncode=returncode,
                            duration_sec=duration_sec,
                            command=cmd,
                            log_path=log_path,
                        )
                    except Exception as exc:
                        result = summarize_failure(
                            model=model,
                            model_label=model_label,
                            mode=mode,
                            run_dir=run_dir,
                            returncode=returncode,
                            duration_sec=duration_sec,
                            status=f"summary failed: {exc}",
                            command=cmd,
                            log_path=log_path,
                        )
                else:
                    result = summarize_failure(
                        model=model,
                        model_label=model_label,
                        mode=mode,
                        run_dir=run_dir,
                        returncode=returncode,
                        duration_sec=duration_sec,
                        status=status,
                        command=cmd,
                        log_path=log_path,
                    )
            results.append(result)
            write_json(root / "summary.json", results)
            write_csv(root / "summary.csv", results)
            run_index += 1


if __name__ == "__main__":
    main()
