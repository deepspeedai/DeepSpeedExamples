# On-Policy Distillation (OPSD) on DeepSpeed

A DeepSpeed-native port of [HJSang/OPSD_OnPolicyDistillation](https://github.com/HJSang/OPSD_OnPolicyDistillation),
removing the verl dependency and building directly on DeepSpeed primitives
(ZeRO-3, hybrid engine, `deepspeed.initialize`).

On-policy distillation trains a small **student** model to imitate a large
frozen **teacher** on the student's *own* generated rollouts. Each training
step has three phases:

```
┌────────────┐   prompts   ┌──────────────────┐   prompt+response   ┌────────────┐
│ Dataloader │ ──────────▶ │ Student rollout  │ ──────────────────▶ │  Teacher   │
└────────────┘             │ (hybrid engine)  │                     │  forward   │
                           └──────────────────┘                     └─────┬──────┘
                                                                          │ logits → CPU cache
                                                                          ▼
                                                              ┌─────────────────────┐
                                                              │ Student forward +   │
                                                              │ streamed KL / JSD + │
                                                              │ backward / step     │
                                                              └─────────────────────┘
```

Loss = per-token divergence (`forward_kl` | `reverse_kl` | `jsd`) between
student and teacher distributions on the student's generated tokens, chunked
over the sequence axis so the full `[B, T, V]` teacher tensor never
co-resides with the student logits on the training device.

## Layout

```
training/opsd/
├── main.py                            # entry point (deepspeed launcher)
├── trainer.py                         # three-phase OPSD training loop
├── config.py                          # OPSDConfig dataclass (JSON-loaded)
├── teacher.py                         # frozen teacher + TeacherLogitCache
├── losses.py                          # streamed distillation loss (fwd/rev KL, JSD)
├── data.py                            # PromptDataset + left-padded collator
├── utils.py                           # response-mask helpers
├── configs/
│   ├── ds_zero3.json                  # base DeepSpeed ZeRO-3 + hybrid engine
│   ├── opsd_hybrid_engine.json        # production-ish hybrid-engine OPSD config
│   ├── smoke_hybrid.json              # 5-step smoke test with Qwen2.5-0.5B / 1.5B
│   ├── smoke_hybrid_gc.json           # smoke test with CUDA graph capture
│   └── smoke_ds_zero0.json            # ZeRO-0 DeepSpeed config for smoke runs
├── data/
│   └── prompts.jsonl                  # sample math prompts
├── benchmarks/                        # rollout / decode micro-benchmarks
├── scripts/
│   └── train_opsd_hybrid.sh           # launch hybrid-engine training
├── tests/                             # CPU-only unit tests (run with pytest)
└── requirements.txt
```

## Quick start

### Install

This example uses DeepSpeed's `deepspeed.runtime.rollout` module (hybrid
engine + rollout API), which landed on `master` after the 0.19.2 release and
is not yet on PyPI. Until DeepSpeed 0.19.3 is released, install it from
source:

```
pip install git+https://github.com/deepspeedai/DeepSpeed.git@master
pip install transformers>=5.0.0 datasets accelerate
```

Once DeepSpeed 0.19.3 ships, a plain `pip install deepspeed` works and the
source line above can be dropped.

### Hybrid-engine training

```
cd training/opsd
NUM_GPUS=8 bash scripts/train_opsd_hybrid.sh configs/opsd_hybrid_engine.json
```

The hybrid engine path lives entirely within DeepSpeed: the student engine
both trains and generates, sharing weights without a copy step.

### Smoke tests (5 steps, small models)

The `smoke_hybrid.json` config runs on 2 GPUs in a few minutes with Qwen2.5-0.5B
(student) and Qwen2.5-1.5B (teacher), so the full pipeline can be validated
end-to-end before scaling up.

```
cd training/opsd
deepspeed --num_gpus 2 main.py --config configs/smoke_hybrid.json
```

## Unit tests

The CPU-runnable test suite exercises the loss math and teacher caching. Run with:

```
cd training/opsd
python -m pytest tests/ -v
```

## Configuration

`OPSDConfig` is a plain dataclass loaded from JSON (no Hydra). The schema:

```json
{
  "student":    { "model_name_or_path": "...", "dtype": "bfloat16" },
  "teacher":    { "model_name_or_path": "...", "dtype": "bfloat16", "offload_to_cpu": true },
  "rollout":    { "engine": "hybrid_engine", ... },
  "distillation": { "loss_type": "reverse_kl", "temperature": 1.0, "chunk_size": 512 },
  "training":   { "train_batch_size": 8, "learning_rate": 1e-6, ... },
  "data":       { "path": "data/prompts.jsonl", "prompt_field": "prompt" },
  "deepspeed_config": "configs/ds_zero3.json"
}
```

See `configs/opsd_hybrid_engine.json` for a fully-populated example.

### Vocabulary-parallel KL

The default `distillation.loss_backend` is `"dense"`. To keep training and
teacher-cache logits in their actual AutoTP vocabulary shards, opt in with:

```json
{
  "teacher": {"autotp_size": 2, "offload_to_cpu": false},
  "distillation": {
    "loss_backend": "vocab_parallel",
    "loss_type": "reverse_kl",
    "temperature": 1.0,
    "chunk_size": 32
  },
  "rollout": {
    "temperature": 0.0,
    "n_samples_per_prompt": 1,
    "use_graph_capture": false,
    "use_shared_prefill": false
  }
}
```

These fields supplement the existing model, data and training settings.
The DeepSpeed JSON must set the same `tensor_parallel.autotp_size` for the
student and select student ZeRO-0 or ZeRO-2. The entry point enables
`tensor_parallel.vocab_parallel_lm_head` before Engine initialization.
Vocabulary-parallel KL requires the DeepSpeed operators introduced by
[PR #8791](https://github.com/deepspeedai/DeepSpeed/pull/8791).
The dense default explicitly retains full-vocabulary heads on both models,
including tied heads that a HuggingFace TP plan would otherwise shard.

The adapter checks each producer's actual shard ranges, ordered TP members,
physical vocabulary width and tokenizer token-ID semantics. TP peers must
consume identical tokens and masks; DP replicas may use different samples
and lengths. TP=1 keeps the vocabulary local even when DP>1. Teacher logits
move directly from the local shard to a host cache; they are not gathered
and sliced. The cache completes D2H before becoming readable and must not
be used after `free()`.

Greedy eager rollout temporarily gathers only the last position's vocabulary
logits, so generation selects a global token. Training and teacher forwards
retain local logits. The hook preserves the head's Parameter objects and
tied-weight identity, and is removed on both success and failure. The
existing next-token shift, temperature-squared scaling, local response-token
mean and Engine gradient-accumulation scaling remain unchanged.

The supported opt-in route uses forward/reverse KL, single-sample greedy HF
generation, no teacher CPU offload, no pipeline/sequence parallelism and no
compiled or accelerated HybridEngine training. JSD, stochastic rollout,
graph capture and shared prefill are rejected. Frozen Qwen3 teachers can
fail in the dependency's replicated-parameter AutoTP gradient-hook setup;
that initialization issue is separate from the KL adapter.

Run the device integration harnesses from this directory with its modules
on `PYTHONPATH`:

```sh
PYTHONPATH="$PWD" torchrun --standalone --nproc_per_node=2 \
  tests/distributed/run_numeric_and_preflight.py --out artifacts/numeric
PYTHONPATH="$PWD" torchrun --standalone --nproc_per_node=4 \
  tests/distributed/run_trainer_integration.py --tp 2 --dtype bfloat16 \
  --tied --student-zero 2 --vocab-size 67 --empty-mask \
  --phase continuous --checkpoint artifacts/checkpoint --out artifacts/trainer
```

The numeric harness compares the actual CUDA operator with an FP64 dense
oracle and checks uniform rejection of malformed layouts and inconsistent
TP inputs. The trainer harness uses random tiny Qwen2 models and real
Engine optimizers; these tests qualify integration mechanics, rather than
pretrained distillation quality. Its `save` phase performs two optimizer
updates and its `resume` phase loads the checkpoint in a new process before
performing the remaining three. Compare those final parameters with a
separate five-update `continuous` run.

`benchmarks/bench_vocab_parallel_opsd.py` measures a full online training
update or replay of frozen real rollout responses. `upstream_dense` imports
an unchanged upstream OPSD directory supplied by `--upstream-opsd-dir`;
the harness explicitly selects the dependency's documented full-logits
configuration for both producers, including the upstream teacher initializer.
`gathered_reference` uses the same candidate head layout with dense math and
a local-slice gather backward; `vocab_parallel` retains local logits. Replay
excludes generation. Online always generates new responses. The benchmark
checks that `--implementation` matches the configured loss backend before
initializing models: `upstream_dense`/`dense_default` require `dense` (or the
omitted default), while `gathered_reference`/`vocab_parallel` require
`vocab_parallel` to construct the sharded heads. It also creates and checks
the result directory before starting GPU work. Report actual
optimizer updates, unique response tokens counted once per TP group, each
run's workload hash, cache payload and device peak; distinguish these from
the original trainer's unsynchronized diagnostic phase timers.

The Qwen2.5 smoke config uses the same 16 math/instruction prompts as the
recorded short-input runs. Replace the model IDs with pinned local snapshot
paths when reproducing a particular result. The measured snapshots were
`7ae557604adf67be50417f59c2c2f167def9a775` for the student and
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306` for the teacher.

```sh
PYTHONPATH="$PWD" torchrun --standalone --nproc_per_node=2 \
  benchmarks/bench_vocab_parallel_opsd.py \
  --config configs/opsd_vocab_parallel_smoke.json \
  --mode online --implementation vocab_parallel --sdpa-backend math \
  --fixture artifacts/smoke-fixture --warmup-updates 0 --measure-updates 5 \
  --output-jsonl artifacts/smoke.jsonl
```

The benchmark checks the actual FP32 optimizer parameters and rejects any
skipped update. `Engine.global_steps` alone is insufficient, since overflow
can advance that counter without updating weights. Model-weight hashes are
reported separately from optimizer-master hashes.

`--sdpa-backend math` selects PyTorch's public math SDPA backend uniformly
for A, B and C. With Torch 2.12.1+cu130 on H100, the default SDPA selection
produced finite logits and KL logit gradients but NaN model gradients on a
long-input, early-EOS batch; the optimizer skipped the update. The matched
math-backend diagnostic had finite gradients and changed FP32 masters.
This is a measured backend limitation, separate from vocabulary-parallel
KL. The benchmark does not change the application's default attention
backend, and these measurements do not qualify every PyTorch/backend pair.

## Design notes

* **Why CPU-cache the teacher logits?** Holding both student and teacher
  `[B, T, V]` tensors on GPU at once doubles memory pressure. Staging the
  teacher to host between the teacher forward and the student backward halves
  the worst-case GPU footprint of the loss path. The streamed loss
  (`losses.streamed_distillation_loss`) pulls teacher chunks back to GPU
  one sequence slice at a time so the full tensor never re-materialises.
  A teacher forward still materializes its output, and backward may retain
  tensors from multiple loss chunks. Chunk size bounds a copy/temporary
  workspace; it does not establish a bound on full-step device peak memory.

* **Why an abstract `RolloutEngine`?** The ABC keeps the trainer
  engine-agnostic so additional backends can be added without touching the
  training loop. DeepSpeed provides the `HybridEngineRollout` implementation;
  external frameworks may plug in their own.

* **Hybrid engine on Qwen-family models uses a ZeRO-3 fallback** (no
  hybrid-engine inference acceleration), since DeepSpeed's inference policy
  list only covers GPT2/GPT-NeoX/OPT/BLOOM/LLAMA/LLAMA2/InternLM as of 0.15.
  The fallback gathers params via `GatheredParameters` and calls the HF
  model's `generate` directly — correct, just ~3-5x slower than the
  accelerated path.

## Other known limitations

* **Reward-weighted distillation** (OPSD's `opd.reward_beta` knob) is not
  ported. Easy to add: scale `per_tok` by a reward weight in the loss path.
* **GRPO and other on-policy RL recipes** are out of scope. The
  `RolloutEngine` abstraction is reusable, but a GRPO trainer would add its
  own advantage / KL-to-reference logic on top.

## References

* OPSD reference repo: <https://github.com/HJSang/OPSD_OnPolicyDistillation>
* DeepSpeed hybrid engine: `deepspeed/runtime/hybrid_engine.py`
