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

## Design notes

* **Why CPU-cache the teacher logits?** Holding both student and teacher
  `[B, T, V]` tensors on GPU at once doubles memory pressure. Staging the
  teacher to host between the teacher forward and the student backward halves
  the worst-case GPU footprint of the loss path. The streamed loss
  (`losses.streamed_distillation_loss`) pulls teacher chunks back to GPU
  one sequence slice at a time so the full tensor never re-materialises.

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
