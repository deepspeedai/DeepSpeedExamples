
# SuperOffload Fine-Tuning Examples

This directory shows how to fine‑tune popular large language models using [DeepSpeed](https://www.deepspeed.ai/) ZeRO Stage 3 with **SuperOffload**. SuperOffload is an optimized CPU offloading engine for full‑parameter training on emerging “Superchips” (NVIDIA GH200 / GB200, AMD MI300A) that provide very high CPU↔GPU bandwidth. It enables:

* 1× GH200: GPT-OSS-20B, Qwen3-14B, Phi-4
* 2× GH200: Seed-OSS-36B, Qwen3-30B-A3B
* 4× GH200: Llama-70B

With common sequence length and batch size, SuperOffload can deliver up to ~500 TFLOPS on GH200—about 50% higher throughput than ZeRO-Offload.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. No custom model code required

All examples use Hugging Face Transformers and DeepSpeed ZeRO Stage 3, no custom modeling code required.

### 3. Enable SuperOffload (one line)

Add the `super_offload` flag to the `offload_optimizer` block in the ZeRO Stage 3 DeepSpeed config:

```jsonc
"zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
        "device": "cpu",
        "pin_memory": true,
        "ratio": 0.90,
        "super_offload": true,
        "cpuadam_cores_perc": 0.90
    }
}
```

To fall back to ZeRO-Offload, remove `"super_offload": true` (and optionally `cpuadam_cores_perc`).

### 4. Run a fine-tuning script

Fine-tune GPT-OSS-20B (1× GH200):

```bash
bash finetune_gpt-oss-20b_1gpu.sh superoffload
```

Fine-tune Qwen3-14B (1× GH200):

```bash
bash finetune_qwen3-14b_1gpu.sh superoffload
```

Fine-tune Phi-4 (1× GH200):

```bash
bash finetune_phi-4_1gpu.sh superoffload
```

Fine-tune Llama 8B (1× GH200):

```bash
bash finetune_llama-8b_1gpu.sh superoffload
```

Fine-tune Seed-OSS-36B (2× GH200):

```bash
bash finetune_seed-oss-36b_2gpu.sh superoffload
```

Fine-tune Llama 70B (4× GH200):

```bash
bash finetune_llama-70b_4gpu.sh superoffload
```

Switch to ZeRO-Offload by replacing `superoffload` with `zerooffload` in the first argument.

Each script optionally accepts a second argument for batch size (default 4):

```bash
bash finetune_qwen3-14b_1gpu.sh superoffload 8
```

Logs, DeepSpeed configs, and outputs are written beside the script location (e.g. `qwen3-14b_superoffload_output/`).


> If a script is missing for a larger model, copy an existing one, change `MODEL_NAME`, and update output naming.


## Notes

* NUMA Binding is required for efficient training on GH200. Each GPU is paired with a CPU to ensure that the training process is launched on the CPU directly associated with that GPU. This pairing improves affinity, delivering higher CPU–GPU bandwidth and greater throughput. In DeepSpeed, we provide a simple interface to enable NUMA binding: simply add the `--bind_cores_to_rank` flag when launching the DeepSpeed engine. 
* Memory System Resource Partitioning and Monitoring (MPAM) is essential for achieving optimal throughput performance. In SuperOffload, GPU execution is overlapped with CPU-based Adam execution. MPAM helps reduce interference between these two processes, leading to smoother execution and better performance.

## Citation

If you use SuperOffload, please cite:

```bib
@inproceedings{superoffload,
    author = {Xinyu Lian and Masahiro Tanaka and Olatunji Ruwase and Minjia Zhang},
    title = "{SuperOffload: Unleashing the Power of Large-Scale LLM Training on Superchips}",
    year = {2026},
    booktitle = {Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating System (ASPLOS'26)}
}
```
