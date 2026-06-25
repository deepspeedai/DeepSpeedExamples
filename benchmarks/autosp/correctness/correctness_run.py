"""
Runs training for a specific configuration (compile mode, sp_size, dp_size, zero_stage)
and saves per-rank losses to a JSON file.

Reuses the existing run.py training script with temporary config files,
launching via accelerate in the same way as run_autosp.sh.
"""

import argparse
import csv
import json
import os
import re
import socket
import subprocess
import sys
import tempfile


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_host_ip():
    try:
        result = subprocess.run(
            ["hostname", "-i"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip().split()[0]
    except Exception:
        return "127.0.0.1"


def create_ds_config(compile_mode, sp_size, dp_size, zero_stage, batch_size, config_path):
    """Create a DeepSpeed JSON config for the given configuration."""
    total_devices = sp_size * dp_size
    train_batch_size = total_devices // sp_size

    config = {
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": zero_stage},
        "gradient_accumulation_steps": 1,
        "gradient_clipping": "auto",
        "steps_per_print": 2000,
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False,
        "sequence_parallel_size" : sp_size
    }
    if compile_mode == "autosp":
        config["compile"] = {
            "deepcompile": True,
            "passes": ["autosp"],
        }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def create_accelerate_config(ds_config_path, num_processes, config_path):
    """Create an accelerate YAML config pointing to the DS JSON config."""
    content = (
        "compute_environment: LOCAL_MACHINE\n"
        "debug: false\n"
        "deepspeed_config:\n"
        "  deepspeed_multinode_launcher: standard\n"
        f"  deepspeed_config_file: {ds_config_path}\n"
        "distributed_type: DEEPSPEED\n"
        "machine_rank: 0\n"
        "main_training_function: main\n"
        "num_machines: 1\n"
        f"num_processes: {num_processes}\n"
        "rdzv_backend: static\n"
        "same_network: true\n"
        "tpu_env: []\n"
        "tpu_use_cluster: false\n"
        "tpu_use_sudo: false\n"
        "use_cpu: false\n"
    )
    with open(config_path, "w") as f:
        f.write(content)


def parse_losses_from_csv(logs_dir, compile_mode, seq_length, num_processes):
    """Read per-rank loss CSV files written by run.py (full precision)."""
    losses = {}
    for rank in range(num_processes):
        csv_path = os.path.join(
            logs_dir, f"loss_{compile_mode}_seq{seq_length}_rank{rank}.csv"
        )
        if not os.path.exists(csv_path):
            continue
        rank_losses = {}
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rank_losses[str(row["step"])] = float(row["loss"])
        losses[str(rank)] = rank_losses
    return losses


def parse_losses_from_stdout(output):
    """Fallback: parse loss values from the printed training output."""
    losses = {}
    for line in output.split("\n"):
        match = re.search(r"\[Rank (\d+)\].*Step (\d+), Loss: ([\d.]+)", line)
        if match:
            rank, step = match.group(1), match.group(2)
            loss = float(match.group(3))
            losses.setdefault(rank, {})[step] = loss
    return losses


def cleanup_csv_files(logs_dir, compile_mode, seq_length, num_processes):
    """Remove loss CSV files created by run.py during training."""
    for rank in range(num_processes):
        csv_path = os.path.join(
            logs_dir, f"loss_{compile_mode}_seq{seq_length}_rank{rank}.csv"
        )
        try:
            os.remove(csv_path)
        except FileNotFoundError:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Run training and capture per-rank losses"
    )
    parser.add_argument("--compile", choices=["compile", "autosp"], required=True)
    parser.add_argument("--sp-size", type=int, required=True)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--zero-stage", type=int, choices=[0, 1], required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    num_processes = args.sp_size * args.dp_size

    script_dir = os.path.dirname(os.path.abspath(__file__))
    autosp_dir = os.path.abspath(os.path.join(script_dir, ".."))
    run_py = os.path.join(autosp_dir, "run.py")
    logs_dir = os.path.join(autosp_dir, "logs")

    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        ds_config_path = os.path.join(tmpdir, "ds_config.json")
        accel_config_path = os.path.join(tmpdir, "accelerate_config.yaml")

        create_ds_config(
            args.compile, args.sp_size, args.dp_size,
            args.zero_stage, args.batch_size, ds_config_path,
        )
        create_accelerate_config(ds_config_path, num_processes, accel_config_path)

        host_ip = get_host_ip()
        port = get_free_port()

        cmd = [
            "accelerate", "launch",
            "--main_process_ip", host_ip,
            "--main_process_port", str(port),
            "--num_machines", "1",
            "--num_processes", str(num_processes),
            "--machine_rank", "0",
            "--config_file", accel_config_path,
            run_py,
            "--model_name", "meta-llama/Llama-2-7b-chat-hf",
            "--batch_size", str(args.batch_size),
            "--seq_length", str(args.seq_length),
            "--sp_size", str(args.sp_size),
            "--dp_size", str(args.dp_size),
            "--backend", "inductor",
            "--compile", args.compile,
            "--num_layers", str(args.num_layers),
            "--steps", str(args.steps),
            "--deterministic",
        ]

        env = os.environ.copy()
        env["NCCL_DEBUG"] = "WARN"

        output = ""
        stderr_output = ""

        if args.verbose:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=autosp_dir,
                env=env,
            )
            for line in process.stdout:
                output += line
                sys.stdout.write(line)
                sys.stdout.flush()
            process.wait()
            return_code = process.returncode
        else:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=autosp_dir, env=env
            )
            output = result.stdout
            stderr_output = result.stderr
            return_code = result.returncode

        # Save training log for debugging
        log_path = args.output_file.replace(".json", ".log")
        with open(log_path, "w") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {return_code}\n")
            f.write("=" * 60 + "\n")
            f.write(output)
            if stderr_output:
                f.write("\n--- STDERR ---\n")
                f.write(stderr_output)

        if return_code != 0:
            print(f"  Training failed (exit code {return_code}). See: {log_path}")
            if not args.verbose:
                lines = (output + stderr_output).strip().split("\n")
                for line in lines[-30:]:
                    print(f"    {line}")
            cleanup_csv_files(logs_dir, args.compile, args.seq_length, num_processes)
            sys.exit(1)

    losses = parse_losses_from_csv(
        logs_dir, args.compile, args.seq_length, num_processes
    )
    cleanup_csv_files(logs_dir, args.compile, args.seq_length, num_processes)

    if not losses:
        print("  Warning: CSV loss files not found, falling back to stdout parsing")
        losses = parse_losses_from_stdout(output)

    if not losses:
        print("  Error: No losses found in training output")
        sys.exit(1)

    result_data = {
        "config": {
            "compile": args.compile,
            "sp_size": args.sp_size,
            "dp_size": args.dp_size,
            "zero_stage": args.zero_stage,
            "steps": args.steps,
        },
        "losses": losses,
    }

    with open(args.output_file, "w") as f:
        json.dump(result_data, f, indent=2)

    num_ranks = len(losses)
    num_steps = max(len(v) for v in losses.values())
    print(f"  Losses saved: {num_ranks} rank(s), {num_steps} step(s) -> {args.output_file}")


if __name__ == "__main__":
    main()
