# rdsp: pipeline parallelism on Ray + DeepSpeed

rdsp (`ray_deepspeed_pipeline`) splits a model into pipeline stages and gives each stage its own GPU count and parallel layout. You can split a Hugging Face vision-language model so the vision encoder gets its own stage, and train it from the HF checkpoint without converting it.

![rdsp architecture. The driver cuts the model into stages and sends every GPU its schedule and batch. Each stage has its own group of GPUs, and every GPU is a Ray actor running a DeepSpeed engine. Activations go forward and gradients come back directly between the GPUs of neighbouring stages.](docs/images/architecture.png)

Every stage is a group of Ray actors, one per GPU, each running a stock DeepSpeed engine. Stages pass activations and gradients to each other directly over NCCL; the driver only plans and schedules.

## Results

Qwen3.5 vision-language models on H100 GPUs, trained on CORD-v2. Median time per training step; lower is better.

| Model | GPUs | rdsp | Megatron-Bridge | TorchTitan |
|---|---|---|---|---|
| Qwen3.5-35B-A3B (mixture of experts) | 16, two nodes | **6.38 s** | 8.67 s | 9.11 s |
| Qwen3.5-27B (dense) | 8, one node | **10.66 s** | 15.13 s | 13.16 s |

- On the 35B that is 1.36× Megatron-Bridge's throughput and 1.43× TorchTitan's; on the 27B, 1.42× and 1.23×. With two rows per microbatch, the fastest setting for both pipelines on the 27B, rdsp is 1.05× Megatron-Bridge.
- On MegatronMIMO's own layouts, with one GPU for the vision encoder, rdsp runs at 0.99–1.09× MIMO's throughput.
- rdsp ran the 35B with the vision encoder on its own 4-GPU stage and six 2-GPU language stages, and the 27B as eight 1-GPU stages, each recomputing its own number of layers. Both measured their own cuts. Megatron-Bridge used the layout of NVIDIA's H100 recipe on the 35B and rdsp's eight-stage split on the 27B. TorchTitan sharded over all GPUs, with expert parallelism on the 35B.

One run per configuration. Settings and commands for each number: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md#reproducing-the-blogs-results). Earlier Qwen3.5-2B and 4B results on 8 GPUs: [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md).

## Install

Linux, CUDA GPUs, Python 3.12. On a fresh node:

```bash
bash bench/setup_node.sh
```

This builds `~/venv` with torch, DeepSpeed at the commit every result was measured with, and rdsp, and pulls the container the Megatron comparisons run in ([docs/GETTING_STARTED.md](docs/GETTING_STARTED.md#install)).

Importing `ray_deepspeed_pipeline` doesn't import Ray or DeepSpeed; only the stage actors do.

## Quickstart

```bash
./recipes/qwen35-2b-tp2pp2dp2.sh                  # Qwen3.5-2B on CORD-v2, 8 GPUs
./recipes/qwen35-2b-tp2pp2dp2.sh --steps 200      # extra flags override the recipe
```

`recipes/` has one script per Qwen3.5-2B and 4B layout in [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md), plus a text-only Qwen3 recipe; the 9B to 35B runs are scripts in `bench/`. [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) covers the flags, your own data and the environment variables.

## Python API

```python
import ray, ray_deepspeed_pipeline as rdsp

def count_tokens(labels):            # tokens the loss is computed on
    return int((labels[:, 1:] != -100).sum())

ray.init()
engine, _, _, _ = rdsp.initialize(
    model=model,                    # HF model, built under accelerate.init_empty_weights()
    weights="/path/to/hf/checkpoint",  # each stage loads only its own tensors
    config=ds_config,               # DeepSpeed config dict
    loss_fn=rdsp.TokenMeanLoss(rdsp.next_token_loss_sum, count_tokens),  # mean over the step's tokens
    pipeline_config=rdsp.PipelineConfig(
        stages=2,
        partition=rdsp.BalancedTransformerBlocks(vision_token_ratio=ratio),
        stage_overrides=(rdsp.StageOverride(stage=0, num_gpus=4, tp=2, zero_stage=1),
                         rdsp.StageOverride(stage=1, num_gpus=4, tp=2, zero_stage=1)),
    ),
)
loss = engine.train_batch(data_iter)   # consumes exactly M (inputs, labels) entries
engine.save_checkpoint("ckpt/", tag="step100")
```

If the input embedding and output layer share a weight and land on different stages, untie them first (`train_vl.py --untie-embeddings`); on one stage they stay tied. The docstrings in `src/ray_deepspeed_pipeline/` give each public name's guarantees and errors. `train_vl.py` is a complete training script.

## How it works

- **Planning.** `rdsp.initialize` lowers the config into an immutable plan: where the model is cut, each stage's GPUs and DeepSpeed config, and a hash that binds checkpoints. Every check, including the cluster's GPU count, runs before any actor starts.
- **Stages.** Each stage runs the HF model's own `forward`. Blocks outside the stage pass their input through, and a hook injects what arrived from upstream. No per-architecture code is needed.
- **Step.** The driver hands each stage its 1F1B schedule, which interleaves forward and backward microbatches. Each stage applies its optimizer step only after every stage reports its backward done.
- **Failures.** A failure before any update raises `StepFailed`: weights are unchanged and a retry is exact. A failure during the update raises `PipelinePoisoned` until a checkpoint is loaded.

Vision encoder placement, how the cuts are chosen, memory, and the reasoning behind the design: [docs/ENGINEERING.md](docs/ENGINEERING.md).

## Model coverage

| Builder | Models | Validation |
|---|---|---|
| `HFModelStage` | Any HF model with a `config`: Llama, Mistral, Qwen2/3/3.5, Qwen3-MoE, Mixtral, Gemma 1–3, Phi-3, OLMo2, Granite, Cohere, StarCoder2, StableLM, DeepSeek-V3, GLM-4/4-MoE/5.3, GPT-2, GPT-NeoX, Falcon, Bloom, Mamba, Qwen3-VL, Qwen3.5-VL | Pipeline loss equals the unsplit model's on CPU (`tests/integration/test_hf_families.py`, `tests/integration/test_vl_pipeline.py`). Qwen3, Qwen3-MoE and Qwen3.5-VL validated on GPU. |
| `CausalLMStage` | Llama, Qwen3, Qwen3-MoE | GPU-validated in every layout in `src/ray_deepspeed_pipeline/support_matrix.py` |
| `GenericSequentialStage` | Plain sequential models | CPU |

TP inside a stage uses the model's AutoTP plan. Configs without one get a plan for the standard projection names. Qwen3-VL injects vision features into its first decoder blocks, so its encoder can't be colocated and the first cut must come after them.

## Tests

```bash
pytest -q          # CPU: full runtime on Ray with a stub engine (about 630 tests); GPU tests skip
ruff check .
```

On a node with GPUs and DeepSpeed (`bench/setup_node.sh`), the same command also runs the GPU tests: parity against the unsplit model, checkpoint round trip and rank-kill recovery.

## Docs

| | |
|---|---|
| [GETTING_STARTED](docs/GETTING_STARTED.md) | Install, recipes, flags, environment variables, rerunning the benchmarks |
| [BENCHMARK_RESULTS](docs/BENCHMARK_RESULTS.md) | Qwen3.5-2B and 4B results, per-layout settings, what moved the numbers |
| [ENGINEERING](docs/ENGINEERING.md) | Design |

## Limitations

- **Scale.** Benchmarked on one 8-GPU node and on two 8× H100 nodes. Two nodes is the most it has run on.
- **Schedule.** 1F1B only, no interleaved stages. No global-norm gradient clipping: a nonzero `gradient_clipping` is rejected.
- **Weights.** No parameters tied across stages.
- **Colocated vision.** It matches the unsplit model. On 32 GB GPUs it needs encoder recompute and is slower; on 80 GB H100s it was the fastest Qwen3.5-4B layout.
- **Balanced cuts.** The cost model ignores communication and counts MoE experts in full, not by the active fraction. `ExplicitCuts` overrides it.
- **Checkpoints.** Node-local: they survive actor loss, not node loss. A stage that spans nodes needs shared storage.
