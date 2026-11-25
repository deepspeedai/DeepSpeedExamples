# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Continue training a causal LM on WikiText-103 with or without DeepSpeed subchannel FP8/INT8 quantization.

Every recipe sees the same data in the same order and starts from the same pretrained weights, so validation loss,
step time and peak memory can be compared against the ``bf16`` baseline. See README.md for launch commands.
"""
import argparse
import json
import math
import os
import time

import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

RECIPES = {
    "bf16": None,
    # The feature's defaults: e4m3, 128-wide blocks, nearest rounding forward, stochastic rounding for dx and dw.
    "e4m3": {},
    # One scale per operand row (per expert in dw): the fast FP8 path.
    "e4m3_row": {name: {
        "block_size": 0,
        "stochastic_rounding": name != "forward"
    }
                 for name in ("forward", "dx", "dw")},
    "int8": {name: {
        "format": "int8",
        "stochastic_rounding": name != "forward"
    }
             for name in ("forward", "dx", "dw")},
    "int8_h128": {
        name: {
            "format": "int8",
            "hadamard_size": 128,
            "stochastic_rounding": name != "forward"
        }
        for name in ("forward", "dx", "dw")
    },
}


def token_blocks(tokenizer, split, seq_len, max_blocks):
    text = "\n\n".join(row for row in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)["text"][:60000]
                       if row.strip())
    ids = tokenizer(text[:max_blocks * seq_len * 8], return_tensors="pt").input_ids[0]
    count = min(max_blocks, ids.numel() // seq_len)
    return ids[:count * seq_len].view(count, seq_len)


def global_mean(value, device):
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor)
    return tensor.item() / dist.get_world_size()


def evaluate(engine, blocks, batch_size):
    engine.eval()
    rank, world = dist.get_rank(), dist.get_world_size()
    local = blocks[rank::world]
    total = 0.0
    with torch.no_grad():
        for start in range(0, local.shape[0], batch_size):
            ids = local[start:start + batch_size].to(engine.device)
            total += engine(input_ids=ids, labels=ids).loss.item() * ids.shape[0]
    engine.train()
    return global_mean(total / local.shape[0], engine.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", choices=RECIPES, required=True)
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_blocks", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bf16_states", action="store_true", help="bf16 master weights and optimizer states")
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--dense", action="store_true", help="not an MoE model: plain data parallelism, no AutoEP")
    parser.add_argument("--grad_checkpointing", action="store_true", help="trade a second forward for memory")
    args = parser.parse_args()

    torch.manual_seed(0)
    world = int(os.environ.get("WORLD_SIZE", 1))
    config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": args.zero_stage
        },
        "gradient_clipping": 1.0,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "torch_adam": True
            }
        },
        "expert_parallel": {
            "enabled": True,
            "autoep_size": world
        },
    }
    if args.dense:
        del config["expert_parallel"]
    if args.bf16_states:
        config["bf16"].update(bf16_master_weights_and_grads=True, bf16_optimizer_states=True)
    if RECIPES[args.recipe] is not None:
        config["quantization"] = {"subchannel": {"enabled": True, **RECIPES[args.recipe]}}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
    engine, _, _, _ = deepspeed.initialize(model=model, config=config)
    rank = dist.get_rank()

    train = token_blocks(tokenizer, "train", args.seq_len, args.steps * args.batch_size * world)
    val = token_blocks(tokenizer, "validation", args.seq_len, args.eval_blocks)

    record = {"recipe": args.recipe, "args": vars(args), "world_size": world, "train_loss": [], "eval": {}, "step_ms": []}
    record["eval"][0] = evaluate(engine, val, args.batch_size)
    if rank == 0:
        print(f"[{args.recipe}] step 0 eval loss {record['eval'][0]:.4f} ppl {math.exp(record['eval'][0]):.3f}", flush=True)

    get_accelerator().reset_peak_memory_stats()
    for step in range(args.steps):
        first = (step * world + rank) * args.batch_size
        ids = train[first:first + args.batch_size].to(engine.device)
        get_accelerator().synchronize()
        start = time.perf_counter()
        loss = engine(input_ids=ids, labels=ids).loss
        engine.backward(loss)
        engine.step()
        get_accelerator().synchronize()
        record["step_ms"].append((time.perf_counter() - start) * 1e3)
        record["train_loss"].append(global_mean(loss.item(), engine.device))
        if rank == 0 and (step + 1) % 20 == 0:
            recent = record["train_loss"][-20:]
            print(f"[{args.recipe}] step {step + 1} train loss {sum(recent) / len(recent):.4f} "
                  f"step {sorted(record['step_ms'][-20:])[10]:.0f} ms", flush=True)
        if (step + 1) % args.eval_every == 0:
            record["eval"][step + 1] = evaluate(engine, val, args.batch_size)
            if rank == 0:
                loss_value = record["eval"][step + 1]
                print(f"[{args.recipe}] step {step + 1} eval loss {loss_value:.4f} ppl {math.exp(loss_value):.3f}",
                      flush=True)

    record["peak_mem_gib"] = get_accelerator().max_memory_allocated() / 2**30
    steady = sorted(record["step_ms"][min(10, len(record["step_ms"]) // 2):])
    record["step_ms_p50"] = steady[len(steady) // 2]
    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(record, handle)
        print(f"[{args.recipe}] done: p50 step {record['step_ms_p50']:.0f} ms, peak {record['peak_mem_gib']:.1f} GiB",
              flush=True)


if __name__ == "__main__":
    main()
