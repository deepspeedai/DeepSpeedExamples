import re
from collections import defaultdict
import pandas as pd
from tabulate import tabulate

def parse_log_file(log_file_path):
    with open(log_file_path, 'r') as f:
        lines = f.readlines()

    # Regex patterns
    trial_header_re = re.compile(
        r"\[Trial (\d+)] pin_memory=(\d), topk=([\d.]+), update=(\d+), overlap_step=(\d+) \(MASTER_PORT=\d+\)"
    )
    time_metrics_re = re.compile(r"\|\s*([^:|]+):\s*([\d.]+)")

    trials = []
    current_config = None
    current_step_metrics = []

    def finalize_trial():
        if current_config and current_step_metrics:
            # Get all unique keys
            all_keys = set()
            for step in current_step_metrics:
                all_keys.update(step.keys())
            # Aggregate and average
            agg = {k: 0.0 for k in all_keys}
            for step in current_step_metrics:
                for k in all_keys:
                    agg[k] += step.get(k, 0.0)
            avg = {f"avg_{k}": agg[k] / len(current_step_metrics) for k in all_keys}
            trials.append({**current_config, **avg, "num_steps": len(current_step_metrics)})

    for line in lines:
        header_match = trial_header_re.search(line)
        if header_match:
            finalize_trial()
            trial_id, pin_memory, topk, update, overlap = header_match.groups()
            current_config = {
                "trial": int(trial_id),
                "pin_memory": bool(int(pin_memory)),
                "topk_ratio": float(topk),
                "update_interval": int(update),
                "overlap_step": bool(int(overlap))
            }
            current_step_metrics = []
            continue

        if "[Rank 0]" in line and "time (ms)" in line:
            metrics = {k.strip(): float(v) for k, v in time_metrics_re.findall(line)}
            current_step_metrics.append(metrics)

    finalize_trial()
    return pd.DataFrame(trials)

if __name__ == "__main__":

    log_file = "zf_benchmark.log"
    df = parse_log_file(log_file)
    df = df.sort_values(by=["topk_ratio", "overlap_step", "update_interval",  "pin_memory"])
    cols_to_display = [
        "trial", "topk_ratio", "update_interval", "overlap_step", "pin_memory",
        "avg_step", "avg_bwd", "avg_fwd", "avg_selective_optimizer_step"
    ]

    headers_with_units = {
        "trial": "trial",
        "pin_memory": "pin_memory",
        "update_interval": "update_interval",
        "overlap_step": "overlap_step",
        "topk_ratio": "topk_ratio",
        "avg_step": "avg_step (ms)",
        "avg_bwd": "avg_bwd (ms)",
        "avg_fwd": "avg_fwd (ms)",
        "avg_selective_optimizer_step": "avg_selective_optimizer_step (ms)"

    }
    headers = [headers_with_units[col] for col in cols_to_display]
    print(tabulate(df[cols_to_display], headers=headers, tablefmt="psql", showindex=False))
