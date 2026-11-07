# Multi-Turn GRPO with Tool Calling (ReTool)

This example trains Qwen3 with GRPO to solve math problems while calling a Python interpreter. In each trajectory the model can write code, read the interpreter's output and continue reasoning, for several turns, before giving its answer.

Rollout and training share one DeepSpeed **hybrid engine** (ZeRO-2): the policy generates with its own training weights, so there is no separate inference server and no weight sync. The example is built on `deepspeed.runtime.rollout`, which provides the multi-turn rollout, the tool interface and the GRPO loss.

## How it works

Each training step has three phases:

1. **Rollout.** Each rank takes its share of the step's prompts and samples `n_samples_per_prompt` trajectories per prompt with `HybridEngineRollout`. When the model writes a `<tool_call>` block, the rollout runs it in the sandboxed `code_interpreter` (`tools.py`) and appends the output as a tool message, for up to `max_turns` turns. With continuous batching, each turn takes a free decode row and releases it when the turn ends, and tools run while the other rows keep decoding.
2. **Reward.** A DAPO strict-box check (`reward.py`) scores each trajectory +1 if the last `\boxed{...}` of the response equals the label, and −1 otherwise. `group_advantages` normalizes each reward within its prompt's group.
3. **Update.** `ResponseTokenBatch.from_rollout` turns the trajectories into training targets with a mask that keeps only the tokens the model generated, so prompts, padding and tool output are not trained on. `grpo_policy_loss` computes PPO-clip with clip-higher (0.2 / 0.28) and no KL term. The loss is averaged over every generated token across ranks, or per trajectory first with `algorithm.loss_aggregation: "sample_mean"`, and the step takes one optimizer update, accumulated over micro-batches.

With one update per rollout, the old policy is the current one, so the old log-probs are the current ones detached and the importance ratio is 1. `rollout.return_log_probs` records them during the rollout and takes them from there instead, through `old_log_probs_from_rollout`.

## Layout

| Path | Contents |
|---|---|
| `main.py` | Entry point: builds the model, the hybrid engine, the rollout and the data, then trains |
| `trainer.py` | The training loop: rollout, reward, update, evaluation, logging and checkpoints |
| `config.py` | The configuration dataclasses and their checks |
| `policy.py` | The policy as DeepSpeed trains it: token log-probs from a chunked LM head |
| `tools.py` | The sandboxed Python interpreter the model calls |
| `reward.py` | The strict-box answer check |
| `data.py` | JSONL prompt loading |
| `configs/` | The three training configs |
| `scripts/` | Data preparation and one launch script per config |

## Prerequisites

- **DeepSpeed with the multi-turn rollout.** `deepspeed.runtime.rollout` with multi-turn tool calling, continuous batching and the GRPO loss is not in a DeepSpeed release yet; it is stage 1 of [RFC #8942](https://github.com/deepspeedai/DeepSpeed/issues/8942). Install DeepSpeed from source with that code.
- **Python packages.** `transformers` 5.0 or later, `numpy`, and `huggingface_hub` for the `hf` download tool:

  ```bash
  pip install "transformers>=5.0.0" numpy huggingface_hub
  ```

- **GPUs.** The smoke test runs on one GPU. The single-GPU config is sized for one 80 GB GPU, and also needs about 64 GB of host memory for the optimizer state it keeps on the CPU. The distributed config is sized for 8 GPUs with 80 GB each.

## Data

The training set is `dapo-math-17k` and the evaluation set is `aime-2024`, in JSONL with a chat-message `prompt` and a string `label`. Download both into `data/`:

```bash
cd training/grpo
bash scripts/prepare_data.sh
```

The script also wraps each AIME prompt in the instructions every `dapo-math-17k` prompt has. The downloaded problems don't ask for a `\boxed{}` answer, which is the only form the scorer accepts, so without them nearly every sample scores −1.

## Run

```bash
cd training/grpo

# smoke test: Qwen3-0.6B, three steps on one GPU
bash scripts/run_smoke_test.sh

# a fine-tuned Qwen3-4B on one GPU
bash scripts/run_single_gpu.sh

# Qwen3-8B on 8 GPUs
bash scripts/run_8gpu.sh
```

Each script runs `deepspeed --num_gpus <N> main.py --config configs/<config>.json`.

Rank 0 prints the loss and the mean reward after each step. When `eval.path` is set, it also prints the AIME accuracy, over every sample, before training, every `eval.interval` steps and at the end. Checkpoints go to `training.save_dir`: a resumable DeepSpeed checkpoint plus a Hugging Face copy in `hf_step_<N>`. To continue a run, set `training.resume_from` to its `save_dir`.

## Configurations

| | `smoke_qwen3_0.6b.json` | `qwen3_4b_sft_1gpu.json` | `qwen3_8b_8gpu.json` |
|---|---|---|---|
| Model | `Qwen/Qwen3-0.6B` | `font-info/qwen3-4b-sft-SGLang-RL` | `Qwen/Qwen3-8B` |
| GPUs | 1 | 1 | 8 |
| Steps | 3 | 500 | 500 |
| Trajectories per step | 8 (2 prompts × 4) | 64 (8 prompts × 8) | 256 (32 prompts × 8) |
| Trajectories per micro-batch, all ranks | 2 | 2 | 8 |
| Turns, context, tokens per turn | 4, 4096, 1024 | 16, 8192, 4096 | 16, 8192, 4096 |
| Continuous batching rows per GPU | 8 | 32 | 16 |
| ZeRO-2 | optimizer on the GPU | optimizer state on the CPU | optimizer on the GPU |
| AIME evaluation | off | every 20 steps | every 20 steps |
| Thinking mode | off | the model's template | off |

All three turn on every rollout optimization that works together: CUDA-graph decode (`use_graph_capture`), and multi-turn continuous batching (`continuous_batch_size`). Training uses bf16, gradient checkpointing, the chunked LM head, SDPA attention, and ZeRO-2 with overlapped communication.

Each config holds the DeepSpeed config inline under `deepspeed_config`, which can also be the path of a DeepSpeed JSON. The trainer fills in `train_micro_batch_size_per_gpu` and `gradient_accumulation_steps`, so that one step covers all of a rank's trajectories.

The main settings:

- `rollout.max_new_tokens`: generated tokens per trajectory, **summed over its turns**.
- `rollout.max_new_tokens_per_turn`: an additional cap on each turn. `eval` has the same two fields.
- `rollout.max_context_len`: total prompt, response and tool tokens per trajectory. A trajectory that reaches it stops as truncated.
- `rollout.continuous_batch_size`: decode rows per GPU for continuous batching; leave it unset for static batching. Rows are sized by `max_context_len`, so the KV cache holds `continuous_batch_size × max_context_len` tokens, about 144 KiB per token for Qwen3-4B and Qwen3-8B. Lower it if rollout runs out of memory.
- `rollout.return_log_probs`: record each generated token's log-prob during the rollout and use them as the PPO old log-probs, off by default. It applies to evaluation rollouts too.
- `rollout.generate_batch_size`: trajectories per `generate` call, in whole prompts; by default a rank's whole share of the step.
- `rollout.chat_template_kwargs`: extra arguments for the chat template, such as `{"enable_thinking": false}` for Qwen3.
- `rollout.repetition_penalty`: 1.0 by default; set it to null to use the model's `generation_config`.
- `training.global_batch_size`: trajectories per step across all ranks, a multiple of ranks × `n_samples_per_prompt` so that each rank rolls out whole prompts.
- `training.micro_batch_size`: trajectories per forward and backward pass across all ranks, split evenly between them. It must divide `global_batch_size`; gradient accumulation covers the rest, so it changes memory and speed, not the update.
- `tool.timeout`, `tool.max_output_chars`: limits on each interpreter call; long output is cut so that one large print cannot fill the context.
- `algorithm.tool_call_bonus`: an optional reward bonus for calling tools on wrong answers, off by default.
- `algorithm.loss_aggregation`: `"token_mean"` (default) weighs every generated token of the step equally; `"sample_mean"` averages each trajectory over its own tokens first, then over trajectories.

## Notes and limitations

- **The sandbox only screens obvious misuse.** It applies an import allowlist, a pattern filter, a memory limit and a timeout. It is not a security boundary, so run untrusted training on isolated machines.
- **ZeRO-2 only.** Multi-turn continuous batching does not support ZeRO-3, whose parameter gathers cannot run inside a captured graph or across rows that finish at different steps.
- **Hybrid engine without injected kernels.** DeepSpeed has no inference policy for Qwen3, so generation runs the Hugging Face model on the training weights, with DeepSpeed's own static cache, graph capture and continuous batching around it.

## Acknowledgement

This example is adapted from the [miles retool_v2](https://github.com/radixark/miles/tree/main/examples/retool_v2) recipe (Copyright 2025-2026 RadixArk, Inc., Apache License 2.0). The code interpreter in `tools.py` and the answer check in `reward.py` are based on miles. The answer check is in turn adapted from verl and EleutherAI's lm-evaluation-harness, both under the Apache License 2.0.
