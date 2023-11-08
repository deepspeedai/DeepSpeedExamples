# Benchmarking Scripts for DeepSpeed-FastGen

## Run the Benchmark

The benchmarking scripts use DeepSpeed-FastGen in the persistent mode.
You can start the server with the command below:

```bash
python server.py [options] start
```

Use the -h option to view all available options. To stop the server, use this command:

```bash
python server.py stop
```

Once the server is up and running, initiate the client using the command below. The -h option will display all the possible options.

```bash
python run_benchmark_client.py [options]
```

The run_all.sh script performs benchmarks across various model sizes and client numbers. For VLLM benchmarks, use the run_all_vllm.sh script. Results are logged in a directory named logs.[BENCHMARK_PARAMETERS].

## Analyze the Benchmark Results

The scripts mentioned below were used for generating the plots featured in our blog. Specify the root directory for log files using --log_dir.

- `plot_th_lat.py`: This script generates charts for throughput and latency across different model sizes and client counts.
- `plot_effective_throughput.py`: Use this to chart effective throughput.
- `plot_latency_percentile.py`: This script will plot the 50th, 90th, and 95th percentile latencies.
