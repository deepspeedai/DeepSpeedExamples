import glob
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import numpy as np

from postprocess_results import read_json, get_summary

bs = 768
    
tp_sizes = {
    "7b": [1],
    "70b": [4, 8],
}

prompt_gen_pairs = [
    (1200, 60),
    (1200, 128),
    (2600, 60),
    (2600, 128),
]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=Path, default="logs.release")
    parser.add_argument("--out_dir", type=Path, default="charts/throughput_latency")
    args = parser.parse_args()
    return args


def extract_values(file_pattern):
    files = glob.glob(file_pattern)

    print(f"Found {len(files)}")
    print('\n'.join(files))

    clients = []
    throughputs = []
    latencies = []
    for f in files:
        prof_args, response_details = read_json(f)
        summary = get_summary(prof_args, response_details)
        clients.append(prof_args["client_num"])
        throughputs.append(summary.throughput)
        latencies.append(summary.latency)

    return clients, throughputs, latencies


def output_charts(model_size, tp, bs, prompt, gen, log_dir, out_dir):
    if not log_dir.exists():
        print(f"Log directory {log_dir} does not exist")
        return
    
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    mii_file_pattern = f"{log_dir}/logs.llama2-{model_size}-tp{tp}-b{bs}/llama2-{model_size}-tp{tp}-b{bs}_c*_p{prompt}_g{gen}.json"
    vllm_file_pattern = f"{log_dir}/logs.vllm-llama2-{model_size}-tp{tp}/vllm-llama2-{model_size}-tp{tp}_c*_p{prompt}_g{gen}.json"

    _, mii_throughputs, mii_latencies = extract_values(mii_file_pattern)
    _, vllm_throughputs, vllm_latencies = extract_values(vllm_file_pattern)

    # Plotting the scatter plot
    plt.figure(figsize=(6, 4))
    
    plt.scatter(vllm_throughputs, vllm_latencies, label=f"vLLM", marker="x", color="orange")
    fit_vllm_x_list = np.arange(min(vllm_throughputs), max(vllm_throughputs), 0.01)
    vllm_vllm_model = np.polyfit(vllm_throughputs, vllm_latencies, 3)
    vllm_model_fn = np.poly1d(vllm_vllm_model)
    plt.plot(fit_vllm_x_list, vllm_model_fn(fit_vllm_x_list), color="orange", alpha=0.5, linestyle="--")

    plt.scatter(mii_throughputs, mii_latencies, label=f"DeepSpeed FastGen", marker="o", color="blue")
    fit_mii_x_list = np.arange(min(mii_throughputs), max(mii_throughputs), 0.01)
    mii_fit_model = np.polyfit(mii_throughputs, mii_latencies, 3)
    mii_model_fn = np.poly1d(mii_fit_model)
    plt.plot(fit_mii_x_list, mii_model_fn(fit_mii_x_list), color="blue", alpha=0.5, linestyle="--")

    plt.title(f'Model Llama 2 {model_size.upper()}, Prompt: {prompt}, Generation: {gen}, TP: {tp}')
    plt.xlabel('Throughput (queries/s)', fontsize=14)
    plt.ylabel('Latency', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    out_file = out_dir / f"th_lat_curve_llama{model_size}_tp{tp}_p{prompt}g{gen}.png"
    print(f"Saving {out_file}")
    plt.savefig(out_file)


if __name__ == "__main__":
    args = get_args()
        
    for model_size, tps in tp_sizes.items():
        for tp in tps:
            for prompt, gen in prompt_gen_pairs:
                output_charts(model_size, tp, bs, prompt, gen, args.log_dir, args.out_dir)

