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

## Running an End-to-End Example

To quickly experience the end-to-end process of running our benchmark and getting results, you can use the `run_example.sh`. This script is designed to execute the benchmark with a specific configuration. The plots below will be generated in the charts directory. These plots show the performance as depicted in figure 8 of our blog [post.](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen#f-other-hardware-platforms)
	   
```bash
bash run_example.sh
```

<div align="center">
  <img src="A6000_benchmarks_example.PNG" alt="" width="800"/><br>

  *Figure 1: Throughput-latency curve and effective throughput of Llama 2 7b using A6000. Runs the client with 60 generation steps and input prompt length of 2600.*<br>
</div>