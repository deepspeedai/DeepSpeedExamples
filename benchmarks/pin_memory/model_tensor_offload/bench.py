# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Model-tensor CPU offload: pinned vs unpinned host memory (ZeRO stage 1/2/3).

Default is ZeRO-3 with offload_optimizer + offload_param. Stages 1 and 2
offload the optimizer only. Pinned arms use DS_PIN_MEMORY_BACKEND=native.
Pass --ablate-register for registered vs unregistered (CUDA).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_PIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PIN_ROOT not in sys.path:
    sys.path.insert(0, _PIN_ROOT)

from common import dist_env, patch_cpp_extension_drop_cxx17, print_arm_result, print_driver_result, run_arm_subprocess


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",
                        type=str,
                        default=None,
                        help="HF model id; architecture from config, random weights. Omit for synthetic.")
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--zero-stage", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--pin", type=int, default=None, help="internal: offload pin_memory 0/1")
    parser.add_argument("--register", type=int, default=None, help="internal: DS_PIN_MEMORY_REGISTER_DEVICE")
    parser.add_argument("--ablate-register", action="store_true")
    return parser.parse_args()


def _build_synthetic(hidden, layers):
    import torch

    class Block(torch.nn.Module):

        def __init__(self, h):
            super().__init__()
            self.fc1 = torch.nn.Linear(h, 4 * h)
            self.fc2 = torch.nn.Linear(4 * h, h)

        def forward(self, x):
            return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

    class Net(torch.nn.Module):

        def __init__(self, h, n):
            super().__init__()
            self.emb = torch.nn.Embedding(32000, h)
            self.blocks = torch.nn.ModuleList([Block(h) for _ in range(n)])
            self.head = torch.nn.Linear(h, 32000, bias=False)

        def forward(self, idx):
            x = self.emb(idx)
            for block in self.blocks:
                x = x + block(x)
            return self.head(x).sum()

    return Net(hidden, layers), 32000


def _zero_config(stage, pin):
    offload_optimizer = {"device": "cpu", "pin_memory": bool(pin)}
    cfg = {"stage": stage, "offload_optimizer": offload_optimizer}
    if stage == 3:
        cfg["offload_param"] = {"device": "cpu", "pin_memory": bool(pin)}
    return cfg


def run_arm(args):
    os.environ["DS_PIN_MEMORY_BACKEND"] = "native"
    os.environ["DS_PIN_MEMORY_REGISTER_DEVICE"] = str(args.register if args.pin else 0)
    dist_env()
    patch_cpp_extension_drop_cxx17()

    import torch
    import deepspeed

    if args.model:
        from transformers import AutoConfig, AutoModelForCausalLM
        config = AutoConfig.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_config(config)
        vocab = model.get_input_embeddings().weight.shape[0]

        def forward_loss(engine, ids):
            return engine(ids).logits.sum()
    else:
        model, vocab = _build_synthetic(args.hidden, args.layers)

        def forward_loss(engine, ids):
            return engine(ids)

    n_params = sum(p.numel() for p in model.parameters())
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch,
        "gradient_accumulation_steps": 1,
        "bf16": {
            "enabled": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4
            }
        },
        "zero_optimization": _zero_config(args.zero_stage, args.pin),
    }

    engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
    dev = engine.device
    dev_api = getattr(torch, dev.type)
    ids = torch.randint(0, vocab, (args.batch, args.seq), dtype=torch.long, device=dev)

    step_times = []
    for step in range(args.warmup + args.steps):
        dev_api.synchronize()
        t0 = time.perf_counter()
        loss = forward_loss(engine, ids)
        engine.backward(loss)
        engine.step()
        dev_api.synchronize()
        if step >= args.warmup:
            step_times.append(time.perf_counter() - t0)

    def _mem(fn):
        try:
            return round(fn() / 1e9, 2)
        except Exception:
            return None

    print_arm_result({
        "experiment": "model_tensor_offload",
        "zero_stage": args.zero_stage,
        "pin_memory": bool(args.pin),
        "register": bool(args.register) if args.pin else None,
        "device": dev.type,
        "model": args.model or f"synthetic-h{args.hidden}-l{args.layers}",
        "params_b": round(n_params / 1e9, 3),
        "batch": args.batch,
        "seq": args.seq,
        "tokens_per_step": args.batch * args.seq,
        "steps": len(step_times),
        "step_avg_s": sum(step_times) / len(step_times),
        "step_min_s": min(step_times),
        "gpu_peak_gb": _mem(dev_api.max_memory_allocated),
    })


def _passthrough(args):
    extra = [
        "--zero-stage",
        str(args.zero_stage),
        "--hidden",
        str(args.hidden),
        "--layers",
        str(args.layers),
        "--batch",
        str(args.batch),
        "--seq",
        str(args.seq),
        "--steps",
        str(args.steps),
        "--warmup",
        str(args.warmup),
    ]
    if args.model:
        extra.extend(["--model", args.model])
    return extra


def run_driver(args):
    if args.ablate_register:
        arms = [("unpinned", 0, 0), ("pinned-unregistered", 1, 0), ("pinned-registered", 1, 1)]
    else:
        arms = [("unpinned", 0, 0), ("pinned", 1, 1)]
    script = os.path.abspath(__file__)
    results = {}
    for name, pin, reg in arms:
        extra = _passthrough(args) + ["--pin", str(pin), "--register", str(reg)]
        results[name] = run_arm_subprocess(script, extra)

    unpinned = results["unpinned"]
    pinned = results["pinned-registered"] if args.ablate_register else results["pinned"]
    print("\n================ Model-tensor CPU-offload step time ================")
    print(f"stage: {pinned['zero_stage']}  model: {pinned['model']}  params: {pinned['params_b']}B  "
          f"device: {pinned['device']}")
    print(f"batch: {pinned['batch']} x seq: {pinned['seq']}  ({pinned['tokens_per_step']} tokens/step)")
    print()
    print(f"{'arm':<22}{'avg step (s)':>14}{'min step (s)':>14}{'GPU peak (GB)':>16}")
    for name, _, _ in arms:
        row = results[name]
        print(f"{name:<22}{row['step_avg_s']:>14.3f}{row['step_min_s']:>14.3f}"
              f"{(row['gpu_peak_gb'] or 0):>16.2f}")
    speedup = unpinned["step_avg_s"] / pinned["step_avg_s"]
    saved = unpinned["step_avg_s"] - pinned["step_avg_s"]
    tok_pin = pinned["tokens_per_step"] / pinned["step_avg_s"]
    tok_unpin = unpinned["tokens_per_step"] / unpinned["step_avg_s"]
    print()
    print(f"pinning speedup: {speedup:.2f}x   saved: {saved:.3f} s/step   "
          f"throughput: {tok_pin:.0f} vs {tok_unpin:.0f} tok/s")
    summary = {"unpinned": unpinned, "pinned": pinned, "speedup": speedup}
    if args.ablate_register:
        summary["pinned-unregistered"] = results["pinned-unregistered"]
    print_driver_result(summary)


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.pin is None:
        run_driver(parsed)
    else:
        run_arm(parsed)
