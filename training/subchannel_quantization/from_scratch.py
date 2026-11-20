# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Train a wide Llama-architecture LM from scratch on WikiText-103 to convergence, with or without subchannel FP8.

For a given seed, every recipe starts from the same initial weights and sees the same data order, so final validation
and test loss can be compared against the ``bf16`` baseline. Run ``python from_scratch.py --prep`` once (single
process) to tokenize the dataset, then launch with torchrun. See README.md.
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

RECIPES = {
    "bf16": None,
    "e4m3": {},
    "e4m3_row": {
        name: {
            "block_size": 0,
            "stochastic_rounding": name != "forward"
        }
        for name in ("forward", "dx", "dw")
    },
}


def prep(data_dir):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    os.makedirs(data_dir, exist_ok=True)
    for split in ("train", "validation", "test"):
        rows = load_dataset("Salesforce/wikitext",
                            "wikitext-103-raw-v1",
                            split=split)["text"]
        ids = []
        for start in range(0, len(rows), 20000):
            chunk = [row for row in rows[start:start + 20000] if row.strip()]
            for encoded in tokenizer(chunk)["input_ids"]:
                ids.extend(encoded)
        tensor = torch.tensor(ids, dtype=torch.int32)
        torch.save(tensor, os.path.join(data_dir, f"{split}.pt"))
        print(split, tensor.numel(), "tokens", flush=True)


def blocks_of(data_dir, split, seq_len):
    ids = torch.load(os.path.join(data_dir, f"{split}.pt")).long()
    count = ids.numel() // seq_len
    return ids[:count * seq_len].view(count, seq_len)


def global_mean(value, device):
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor)
    return tensor.item() / dist.get_world_size()


def evaluate(engine, blocks, batch_size):
    engine.eval()
    rank, world = dist.get_rank(), dist.get_world_size()
    usable = blocks[:blocks.shape[0] // world * world]
    local = usable[rank::world]
    total = 0.0
    with torch.no_grad():
        for start in range(0, local.shape[0], batch_size):
            ids = local[start:start + batch_size].to(engine.device)
            total += engine(input_ids=ids,
                            labels=ids).loss.item() * ids.shape[0]
    engine.train()
    return global_mean(total / local.shape[0], engine.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep",
                        action="store_true",
                        help="tokenize WikiText-103 into --data_dir and exit")
    parser.add_argument("--data_dir",
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "data"))
    parser.add_argument("--recipe", choices=RECIPES)
    parser.add_argument("--out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="stop early (timing runs); the schedule is unchanged")
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=11008)
    args = parser.parse_args()
    if args.prep:
        return prep(args.data_dir)
    if args.recipe is None or args.out is None:
        parser.error("--recipe and --out are required for training")

    from transformers import LlamaConfig, LlamaForCausalLM
    world = int(os.environ.get("WORLD_SIZE", 1))
    train = blocks_of(args.data_dir, "train", args.seq_len)
    val = blocks_of(args.data_dir, "validation", args.seq_len)
    test = blocks_of(args.data_dir, "test", args.seq_len)
    steps_per_epoch = train.shape[0] // (world * args.batch_size)
    total_steps = int(args.epochs * steps_per_epoch)

    config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2
        },
        "gradient_clipping": 1.0,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "betas": [0.9, 0.95],
                "weight_decay": 0.1,
                "torch_adam": True
            }
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps": total_steps,
                "warmup_num_steps": args.warmup,
                "warmup_min_ratio": 0.0,
                "cos_min_ratio": 0.0
            }
        },
    }
    if RECIPES[args.recipe] is not None:
        config["quantization"] = {
            "subchannel": {
                "enabled": True,
                **RECIPES[args.recipe]
            }
        }

    torch.manual_seed(args.seed)
    model = LlamaForCausalLM(
        LlamaConfig(vocab_size=50257,
                    hidden_size=args.hidden,
                    intermediate_size=args.intermediate,
                    num_hidden_layers=args.layers,
                    num_attention_heads=args.hidden // 128,
                    num_key_value_heads=args.hidden // 128,
                    max_position_embeddings=args.seq_len,
                    tie_word_embeddings=True,
                    bos_token_id=50256,
                    eos_token_id=50256))
    engine, _, _, scheduler = deepspeed.initialize(model=model, config=config)
    rank = dist.get_rank()
    params = sum(p.numel() for p in model.parameters())

    record = {
        "recipe": args.recipe,
        "args": vars(args),
        "world_size": world,
        "params": params,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "train_loss": [],
        "eval": {},
        "step_ms": []
    }

    def save():
        if rank == 0:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)),
                        exist_ok=True)
            with open(args.out, "w") as handle:
                json.dump(record, handle)

    if rank == 0:
        print(
            f"[{args.recipe}] {params / 1e9:.2f}B params, {steps_per_epoch} steps/epoch, {total_steps} steps",
            flush=True)
    last_step = min(total_steps,
                    args.max_steps) if args.max_steps else total_steps
    window_loss, window_start, perm, epoch = [], None, None, -1
    get_accelerator().reset_peak_memory_stats()
    for step in range(last_step):
        if step // steps_per_epoch != epoch:
            epoch = step // steps_per_epoch
            perm = torch.randperm(
                train.shape[0],
                generator=torch.Generator().manual_seed(1000 * args.seed +
                                                        epoch))
        first = ((step % steps_per_epoch) * world + rank) * args.batch_size
        ids = train[perm[first:first + args.batch_size]].to(engine.device)
        if step % 50 == 0:
            get_accelerator().synchronize()
            now = time.perf_counter()
            if window_start is not None:
                record["step_ms"].append((now - window_start) / 50 * 1e3)
                record["train_loss"].append(
                    global_mean(
                        sum(window_loss) / len(window_loss), engine.device))
                if rank == 0 and step % 250 == 0:
                    print(
                        f"[{args.recipe}] step {step} train {record['train_loss'][-1]:.4f} "
                        f"{record['step_ms'][-1]:.0f} ms/step lr {scheduler.get_last_lr()[0]:.2e}",
                        flush=True)
            window_loss = []
            get_accelerator().synchronize()
            window_start = time.perf_counter()
        loss = engine(input_ids=ids, labels=ids).loss
        engine.backward(loss)
        engine.step()
        window_loss.append(loss.detach())
        if step % 50 == 49:
            window_loss = [float(torch.stack(window_loss).float().mean())]
        if (step + 1) % args.eval_every == 0 or step + 1 == last_step:
            record["eval"][step + 1] = evaluate(engine, val, args.batch_size)
            if rank == 0:
                value = record["eval"][step + 1]
                print(
                    f"[{args.recipe}] step {step + 1} (epoch {(step + 1) / steps_per_epoch:.2f}) val loss {value:.4f} "
                    f"ppl {math.exp(value):.2f}",
                    flush=True)
            save()
            window_start = None
            window_loss = []

    record["test_loss"] = evaluate(engine, test, args.batch_size)
    record["peak_mem_gib"] = get_accelerator().max_memory_allocated() / 2**30
    steady = sorted(record["step_ms"][1:] or record["step_ms"])
    record["step_ms_p50"] = steady[len(steady) // 2] if steady else None
    save()
    if rank == 0:
        print(
            f"[{args.recipe}] done: test loss {record['test_loss']:.4f} ppl {math.exp(record['test_loss']):.2f}, "
            f"p50 step {record['step_ms_p50']} ms, peak {record['peak_mem_gib']:.1f} GiB",
            flush=True)


if __name__ == "__main__":
    main()
