# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
ZeRO-3 CPU-offload end-to-end benchmark: registered vs unregistered native
pinned host memory.

Measures training step time with ZeRO-3, offload_optimizer + offload_param
(both cpu, pin_memory=True) and DS_PIN_MEMORY_BACKEND=native, comparing
DS_PIN_MEMORY_REGISTER_DEVICE=1 vs 0. Works on CUDA and XPU (any accelerator
with native pin + register_host_memory support).

Each arm runs in its own subprocess with a fresh rendezvous port so device
state never leaks between arms.

Examples:
    # real model architecture (config fetched from the HF hub, random weights)
    python zero3_offload_bench.py --model Qwen/Qwen2.5-7B --batch 4 --seq 512

    # synthetic MLP stack, no network needed
    python zero3_offload_bench.py --hidden 2048 --layers 12 --batch 4 --seq 128

Note: the native backend mlocks host memory; raise RLIMIT_MEMLOCK
(ulimit -l) or run as root for multi-GB models.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time


def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "on")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",
                   type=str,
                   default=None,
                   help="HF model id (e.g. Qwen/Qwen2.5-7B); architecture is loaded from "
                   "the config with random weights. Omit to use the synthetic model.")
    p.add_argument("--hidden", type=int, default=2048, help="synthetic model hidden size")
    p.add_argument("--layers", type=int, default=12, help="synthetic model layer count")
    p.add_argument("--batch", type=int, default=4, help="micro batch size per gpu")
    p.add_argument("--seq", type=int, default=128, help="sequence length")
    p.add_argument("--steps", type=int, default=4, help="timed steps per arm")
    p.add_argument("--warmup", type=int, default=2, help="warmup steps per arm")
    p.add_argument("--register",
                   type=int,
                   default=None,
                   help="internal: run a single arm with DS_PIN_MEMORY_REGISTER_DEVICE=0/1")
    return p.parse_args()


def _free_port():
    # A stale listener from an interrupted rank makes the next init hang in a
    # collective, so always rendezvous on a fresh ephemeral port.
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


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
            for b in self.blocks:
                x = x + b(x)
            return self.head(x).sum()

    return Net(hidden, layers), 32000


def run_arm(args):
    """Single arm: one process, one DS_PIN_MEMORY_REGISTER_DEVICE setting."""
    os.environ["DS_PIN_MEMORY_BACKEND"] = "native"
    os.environ["DS_PIN_MEMORY_REGISTER_DEVICE"] = str(args.register)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(_free_port()), RANK="0", WORLD_SIZE="1", LOCAL_RANK="0")

    import torch
    import torch.utils.cpp_extension as _ce

    # torch-nightly requires C++20; SYCL toolchain flags may carry -std=c++17,
    # which (appearing last) downgrades the dialect and breaks torch headers.
    _orig_ce_load = _ce.load

    def _ce_load_without_cxx17(*a, **kw):
        for key in ("extra_cflags", "extra_cxxflags"):
            if kw.get(key):
                kw[key] = [f for f in kw[key] if f != "-std=c++17"]
        return _orig_ce_load(*a, **kw)

    _ce.load = _ce_load_without_cxx17

    import deepspeed

    if args.model:
        from transformers import AutoConfig, AutoModelForCausalLM
        config = AutoConfig.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_config(config)
        vocab = config.vocab_size

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
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
        },
    }

    engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
    dev = engine.device
    # Device API via getattr so the accelerator-agnostic rule (no hardcoded
    # backend namespaces) holds while working on cuda and xpu alike.
    dev_api = getattr(torch, dev.type)
    sync = dev_api.synchronize
    ids = torch.randint(0, vocab, (args.batch, args.seq), dtype=torch.long, device=dev)

    step_times = []
    for step in range(args.warmup + args.steps):
        sync()
        t0 = time.perf_counter()
        loss = forward_loss(engine, ids)
        engine.backward(loss)
        engine.step()
        sync()
        dt = time.perf_counter() - t0
        if step >= args.warmup:
            step_times.append(dt)

    def _mem(fn):
        try:
            return round(fn() / 1e9, 2)
        except Exception:
            return None

    result = {
        "register": bool(args.register),
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
    }
    print("ARMRESULT=" + json.dumps(result), flush=True)


def run_driver(args):
    """Run both arms in subprocesses and print the comparison."""
    results = {}
    for reg in (1, 0):
        cmd = [sys.executable, os.path.abspath(__file__), "--register", str(reg), "--model"] + \
              ([args.model] if args.model else ["None"]) + \
              ["--hidden", str(args.hidden), "--layers", str(args.layers), "--batch", str(args.batch),
               "--seq", str(args.seq), "--steps", str(args.steps), "--warmup", str(args.warmup)]
        # argparse cannot take a literal None for --model; drop it instead.
        if not args.model:
            cmd = cmd[:cmd.index("--model")] + cmd[cmd.index("--model") + 2:]
        print(f"[driver] running arm register={reg} ...", flush=True)
        proc = subprocess.run(cmd, env=os.environ.copy(), capture_output=True, text=True)
        arm = None
        for line in proc.stdout.splitlines():
            if line.startswith("ARMRESULT="):
                arm = json.loads(line[len("ARMRESULT="):])
        if arm is None:
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:], file=sys.stderr)
            raise RuntimeError(f"arm register={reg} produced no result (rc={proc.returncode})")
        results[reg] = arm

    on, off = results[1], results[0]
    print("\n================ ZeRO-3 CPU-offload step time ================")
    print(f"model: {on['model']}  params: {on['params_b']}B  device: {on['device']}")
    print(f"batch: {on['batch']} x seq: {on['seq']}  ({on['tokens_per_step']} tokens/step)")
    print()
    print(f"{'arm':<22}{'avg step (s)':>14}{'min step (s)':>14}{'GPU peak (GB)':>16}")
    print(f"{'registered':<22}{on['step_avg_s']:>14.3f}{on['step_min_s']:>14.3f}"
          f"{(on['gpu_peak_gb'] or 0):>16.2f}")
    print(f"{'unregistered':<22}{off['step_avg_s']:>14.3f}{off['step_min_s']:>14.3f}"
          f"{(off['gpu_peak_gb'] or 0):>16.2f}")
    speedup = off['step_avg_s'] / on['step_avg_s']
    saved = off['step_avg_s'] - on['step_avg_s']
    tok_on = on['tokens_per_step'] / on['step_avg_s']
    tok_off = off['tokens_per_step'] / off['step_avg_s']
    print()
    print(f"speedup: {speedup:.2f}x   saved: {saved:.3f} s/step   "
          f"throughput: {tok_on:.0f} vs {tok_off:.0f} tok/s")
    print(f"DRIVERRESULT={json.dumps({'registered': on, 'unregistered': off, 'speedup': speedup})}", flush=True)


if __name__ == "__main__":
    _args = parse_args()
    if _args.register is None:
        run_driver(_args)
    else:
        run_arm(_args)
