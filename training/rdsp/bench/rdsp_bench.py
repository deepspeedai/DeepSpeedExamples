# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Train one rdsp benchmark cell on the local GPUs and write its results as JSON.

    python rdsp_bench.py --pp 4 --m 16 --seq 512 --data <tokens.pt> --out <result.json>

Assembles the pipeline the same way as the Qwen3-0.6B GPU acceptance test.
--data is a [steps, m, b, s+1] token tensor shared with the Megatron runs.
"""

import argparse
import json
import os
import time

import ray
import torch
import torch.nn.functional as F
import transformers
from bench_hooks import (
    apply_fusions,
    fused_engine_factory,
    fused_prof_engine_factory,
    fused_timed_engine_factory,
    tagged_stage_builder,
    timed_engine_factory,
)

import ray_deepspeed_pipeline as rdsp
from ray_deepspeed_pipeline.compiler import lower
from ray_deepspeed_pipeline.config import StageOverride
from ray_deepspeed_pipeline.coordinator import PipelineCoordinator
from ray_deepspeed_pipeline.engine import RayPipelineEngine
from ray_deepspeed_pipeline.stage_group import create_stage_clients


def loss_fn(out, labels):
    return F.cross_entropy(out.float().reshape(-1, out.shape[-1]), labels.reshape(-1))


p = argparse.ArgumentParser()
for name, typ, default in [
    ("hf", str, "/cache/qwen3-0.6b-untied"),
    ("pp", int, 2),
    ("m", int, 8),
    ("seq", int, 2048),
    ("steps", int, 60),
    ("warmup", int, 10),
    ("data", str, None),
    ("out", str, None),
    ("cuts", str, ""),
    ("last_stage_gpus", int, 1),
    ("timeline", int, 0),
    ("timed", int, 0),
    ("fused", int, 0),
    ("prof", int, 0),
    ("b", int, 0),
]:
    p.add_argument(f"--{name}", type=typ, default=default)
a = p.parse_args()
b = a.b or (4 if a.seq == 512 else 1)  # rows per microbatch: 2048 tokens by default
# a stage with k GPUs splits every microbatch's rows k ways (data parallel
# inside the stage), so the microbatch needs a multiple of k rows
b = max(b, a.last_stage_gpus) if b % a.last_stage_gpus else b
os.environ["RDSP_BENCH_WARMUP"] = str(a.warmup)
if a.timed:  # fresh per-stage timing files for this run
    import shutil

    shutil.rmtree(os.environ.get("RDSP_BENCH_DIR", "/tmp/rdsp_bench"), ignore_errors=True)

ray.init(
    include_dashboard=False,
    log_to_driver=False,
    runtime_env={
        "env_vars": {
            "PYTHONPATH": os.path.dirname(os.path.abspath(__file__)),
            "RDSP_BENCH_WARMUP": str(a.warmup),
        }
    },
)
if a.fused:
    apply_fusions()  # before model construction: fused norm/MLP classes
    from liger_kernel.transformers import LigerCrossEntropyLoss

    _fused_ce = LigerCrossEntropyLoss()

    def loss_fn(out, labels):  # fused CE: no fp32 copy of the [tokens, vocab] logits
        return _fused_ce(out.reshape(-1, out.shape[-1]), labels.reshape(-1))


model = transformers.AutoModelForCausalLM.from_pretrained(a.hf, dtype=torch.bfloat16)
assert model.lm_head.weight is not model.model.embed_tokens.weight  # untied checkpoint

ds = {
    "train_micro_batch_size_per_gpu": b,
    "gradient_accumulation_steps": a.m,
    "train_batch_size": b * a.m,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "torch_adam": True,
            **({"fused": True} if a.fused else {}),
        },
    },
    "zero_optimization": {"stage": 0},
    "bf16": {"enabled": True},
    "gradient_clipping": 0.0,
    "steps_per_print": 10**9,
}
part = (
    rdsp.ExplicitCuts(tuple(int(c) for c in a.cuts.split(",")))
    if a.cuts
    else rdsp.UniformTransformerBlocks()
)
overrides = (
    (StageOverride(stage=a.pp - 1, num_gpus=a.last_stage_gpus),) if a.last_stage_gpus > 1 else ()
)
plan = lower(
    model,
    rdsp.PipelineConfig(stages=a.pp, partition=part, microbatches=a.m, stage_overrides=overrides),
    ds,
)
# Clean throughput runs use rdsp's own engine factory (None); --timed 1 adds
# the CUDA-event proxy (one host sync per step) for per-stage busy times.
if a.fused:
    factory = (
        fused_prof_engine_factory
        if a.prof
        else fused_timed_engine_factory
        if a.timed
        else fused_engine_factory
    )
else:
    factory = timed_engine_factory if a.timed else None
clients = create_stage_clients(
    model, plan, loss_fn, use_gpu=True, stage_builder=tagged_stage_builder, engine_factory=factory
)
engine = RayPipelineEngine(PipelineCoordinator(plan, clients))
del model  # the driver holds no weights

X = torch.load(a.data)  # [steps, m, b, s+1]; labels are the inputs shifted by one


def mbs(step):
    return iter(
        [
            (
                X[step % X.shape[0], k][:, :-1].contiguous(),
                X[step % X.shape[0], k][:, 1:].contiguous(),
            )
            for k in range(a.m)
        ]
    )


times, losses, windows = [], [], []
for step in range(a.steps):
    t0, w0 = time.perf_counter(), time.time()
    losses.append(float(engine.train_batch(data_iter=mbs(step))))
    times.append(time.perf_counter() - t0)
    windows.append((w0, time.time()))  # epoch seconds, to align with the Ray timeline

profile = {}
if a.timeline:
    ray.timeline(filename=a.out + ".timeline.json")
    with open(a.out + ".timeline.json") as f:
        profile["timeline"] = json.load(f)
    profile["step_windows"] = windows
if a.timed:
    import glob

    time.sleep(1)  # let the actors' last JSON lines reach disk
    stage_ops = {}
    bench_dir = os.environ.get("RDSP_BENCH_DIR", "/tmp/rdsp_bench")
    for path in glob.glob(os.path.join(bench_dir, "*.jsonl")):
        with open(path) as f:
            stage_ops[os.path.basename(path)] = [json.loads(line) for line in f]
    profile["stage_ops"] = stage_ops
order = (
    {c.stage: [ray.get(act.executed_commands.remote()) for act in c.actors] for c in clients}
    if a.timeline
    else None
)
with open(a.out, "w") as f:
    json.dump(
        {
            "system": "rdsp",
            "fused": a.fused,
            "pp": a.pp,
            "m": a.m,
            "seq": a.seq,
            "b": b,
            "cuts": a.cuts,
            "last_stage_gpus": a.last_stage_gpus,
            "stage_tags": [f"blocks{s.block_start:02d}-{s.block_stop:02d}" for s in plan.stages],
            "step_s": times[a.warmup :],
            "loss": losses,
            "executed": order,
            "profile": profile,
        },
        f,
    )
ray.shutdown()
