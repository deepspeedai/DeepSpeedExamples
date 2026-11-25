# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Live demo: the rdsp pipeline engine, end to end, on your laptop.

    python demo.py

Runs the real production path (PipelineConfig -> ExecutionPlan -> coordinator
-> Ray actors -> stage adapters) on CPU, with a plain-torch engine stub in the
DeepSpeed slot (the one component that needs CUDA). Everything else is exactly
what runs on GPUs.

It trains a toy LM split across 2 Ray actor stages and checks, every step,
that the pipeline reproduces the monolithic single-process model. Then it
demonstrates the failure contracts. ~30 seconds.

The same path with real DeepSpeed engines and Qwen3-0.6B, on 2 Modal GPUs
(from the repo root):
    modal run scripts/modal_tests.py --gpus L4:2 \
        --tests tests/integration/test_two_stage_baseline.py
"""

import logging
import time

import ray
import torch
import torch.nn as nn
import torch.nn.functional as F

import ray_deepspeed_pipeline as rdsp
from ray_deepspeed_pipeline.compiler import lower
from ray_deepspeed_pipeline.coordinator import PipelineCoordinator
from ray_deepspeed_pipeline.engine import RayPipelineEngine
from ray_deepspeed_pipeline.errors import StepFailed, UnsupportedEngineMethod
from ray_deepspeed_pipeline.schedule import generate_commands, peak_in_flight
from ray_deepspeed_pipeline.stage_group import create_stage_clients

N_MB, ROWS, SEQ, VOCAB, LR, STEPS = 4, 2, 16, 50, 0.05, 5


# --- a tiny transformer-shaped LM (embed -> blocks -> norm -> head) ----------


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 32)

    def forward(self, x):
        return x + torch.tanh(self.fc(x))


class ToyLM(nn.Module):
    def __init__(self, n_blocks=8):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, 32)
        self.blocks = nn.ModuleList(Block() for _ in range(n_blocks))
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, VOCAB, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


# --- stand-in for the stage-local DeepSpeed engine (the CUDA-only component) -


class StubEngine:
    def __init__(self, module, conf):
        self.module = module
        self.optimizer = torch.optim.AdamW(module.parameters(), lr=LR)

    def __call__(self, x):
        return self.module(x)

    def backward(self, loss):
        loss.backward()

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()


def stub_engine_factory(module, conf):
    return StubEngine(module, conf)


def loss_fn(outputs, labels):
    return F.cross_entropy(outputs.reshape(-1, VOCAB), labels.reshape(-1))


def main():
    print(__doc__.split("\n")[0])
    print("=" * 72)

    torch.manual_seed(0)
    model = ToyLM()
    ids = torch.randint(0, VOCAB, (N_MB * ROWS, SEQ))
    labels = torch.randint(0, VOCAB, (N_MB * ROWS, SEQ))

    def mbs():
        return iter(
            [
                (ids[k * ROWS : (k + 1) * ROWS], labels[k * ROWS : (k + 1) * ROWS])
                for k in range(N_MB)
            ]
        )

    # 1. lower the public config to the one internal ExecutionPlan
    cfg = rdsp.PipelineConfig(
        stages=2, partition=rdsp.UniformTransformerBlocks(), microbatches=N_MB
    )
    plan = lower(model, cfg, {"train_batch_size": N_MB * ROWS})
    print(
        f"\n[plan]     {len(plan.stages)} stages | {plan.global_microbatches} "
        f"microbatches | schedule {plan.schedule.kind} | hash {plan.plan_hash()[:12]}…"
    )
    for s in plan.stages:
        print(
            f"           stage {s.index}: blocks [{s.block_start}:{s.block_stop}), "
            f"{len(s.parameter_names)} param tensors, {s.num_gpus} gpu(s)"
        )

    seqs = generate_commands(plan, 1, 0)
    for s, seq in seqs.items():
        trace = " ".join(
            f"{c.kind[0].upper()}{c.microbatch}" for c in seq if c.kind in ("forward", "backward")
        )
        print(f"[1f1b]     stage {s}: {trace}   (peak cached activations: {peak_in_flight(seq)})")

    # 2. real Ray actors, one per stage
    ray.init(
        num_cpus=3, include_dashboard=False, log_to_driver=False, logging_level=logging.WARNING
    )
    clients = create_stage_clients(
        model, plan, loss_fn, engine_factory=stub_engine_factory, use_gpu=False
    )
    engine = RayPipelineEngine(PipelineCoordinator(plan, clients))
    print(
        f"\n[actors]   2 stage actors up; loss_fn on terminal stage only: "
        f"{ray.get(clients[1].actors[0].has_loss_fn.remote())} / "
        f"{ray.get(clients[0].actors[0].has_loss_fn.remote())}"
    )

    # 3. train, checking parity against the monolithic model every step
    ref = ToyLM()
    ref.load_state_dict(model.state_dict())
    opt = torch.optim.AdamW(ref.parameters(), lr=LR)

    print("\n[check]    same data, same starting weights, trained two ways:")
    print("           'pipeline'   = split across the 2 Ray actors above")
    print("           'monolithic' = the unsplit model in this process (the reference)")
    print("           if the pipeline math is right, the columns must match")
    print(f"\n[train]    {'step':>4} {'pipeline loss':>14} {'monolithic':>12} {'|diff|':>10}")
    t0 = time.time()
    for step in range(1, STEPS + 1):
        staged = float(engine.train_batch(data_iter=mbs()))
        graph_loss = loss_fn(ref(ids), labels)
        opt.zero_grad()
        graph_loss.backward()
        opt.step()
        ref_loss = graph_loss.detach()
        diff = abs(staged - float(ref_loss))
        print(f"           {step:>4} {staged:>14.6f} {float(ref_loss):>12.6f} {diff:>10.2e}")
        assert diff < 1e-4, "pipeline diverged from the monolithic model!"
    print(
        f"[train]    {STEPS} global steps in {time.time() - t0:.1f}s, "
        f"global_steps={engine.global_steps} — parity holds"
    )

    # 4. the contracts, demonstrated live
    ev = engine.eval_batch(data_iter=mbs())
    print(
        f"\n[eval]     loss {float(ev):.6f} | global_steps still "
        f"{engine.global_steps} (no update, no step)"
    )

    # deliberately misuse the library and confirm it refuses cleanly
    try:
        engine.train_batch(data_iter=iter([next(mbs())]))  # only 1 of 4 microbatches
    except StepFailed:
        print(
            f"[guard ok] fed 1 of the 4 required microbatches on purpose -> "
            f"clean error, no weights touched (global_steps still {engine.global_steps})"
        )

    try:
        engine.backward(None)
    except UnsupportedEngineMethod:
        print(
            "[guard ok] engine.backward() on purpose -> refused; "
            "train_batch() is the only training entry point"
        )

    ray.shutdown()
    print("\n" + "=" * 72)
    print("Everything above ran the production runtime; only the stage-local")
    print("engine was a torch stub. The same path with real DeepSpeed engines")
    print("and Qwen3-0.6B: see the GPU acceptance run in this file's docstring.")


if __name__ == "__main__":
    main()
