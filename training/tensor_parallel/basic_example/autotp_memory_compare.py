import argparse
from dataclasses import dataclass
import os

import torch
import torch.distributed as dist
import deepspeed
from transformers import AutoModelForCausalLM


@dataclass
class ModelParallelUnit:
    """Minimal MPU for DeepSpeed TP+DP."""

    tp_group: dist.ProcessGroup
    dp_group: dist.ProcessGroup
    tp_size: int
    dp_size: int
    tp_rank: int
    dp_rank: int

    def get_data_parallel_group(self):
        return self.dp_group

    def get_model_parallel_group(self):
        return self.tp_group

    def get_data_parallel_world_size(self):
        return self.dp_size

    def get_model_parallel_world_size(self):
        return self.tp_size

    def get_data_parallel_rank(self):
        return self.dp_rank

    def get_model_parallel_rank(self):
        return self.tp_rank


def parse_args():
    parser = argparse.ArgumentParser(description="Compare AutoTP memory usage between init paths.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Passed by deepspeed/torchrun.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--tp_size", type=int, default=4)
    parser.add_argument("--dp_size", type=int, default=2)
    parser.add_argument("--zero_stage", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--mode",
                        type=str,
                        default="config",
                        choices=["config", "traditional"],
                        help="config = config-driven path, traditional = call tp_model_init")
    return parser.parse_args()


def build_tp_dp_groups(rank, world_size, tp_size, dp_size):
    if tp_size * dp_size != world_size:
        raise ValueError(f"tp_size ({tp_size}) * dp_size ({dp_size}) must equal world_size ({world_size})")

    tp_rank = rank % tp_size
    dp_rank = rank // tp_size

    tp_group = None
    dp_group = None

    for dp_idx in range(dp_size):
        tp_ranks = list(range(dp_idx * tp_size, (dp_idx + 1) * tp_size))
        group = dist.new_group(tp_ranks)
        if rank in tp_ranks:
            tp_group = group

    for tp_idx in range(tp_size):
        dp_ranks = [tp_idx + dp_idx * tp_size for dp_idx in range(dp_size)]
        group = dist.new_group(dp_ranks)
        if rank in dp_ranks:
            dp_group = group

    return tp_group, dp_group, tp_rank, dp_rank


def broadcast_inputs(input_ids, labels, tp_group, tp_src_rank):
    dist.broadcast(input_ids, src=tp_src_rank, group=tp_group)
    dist.broadcast(labels, src=tp_src_rank, group=tp_group)


def get_precision_dtype(precision):
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def summarize(values):
    return min(values), max(values), sum(values) / len(values)


def gather_and_print(tag, device, rank, world_size):
    stats = torch.tensor(
        [torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
        device=device,
    )
    gathered = [torch.zeros_like(stats) for _ in range(world_size)]
    dist.all_gather(gathered, stats)

    if rank == 0:
        allocs = [t[0].item() / (1024**3) for t in gathered]
        reservs = [t[1].item() / (1024**3) for t in gathered]
        alloc_min, alloc_max, alloc_mean = summarize(allocs)
        res_min, res_max, res_mean = summarize(reservs)
        print(f"[MEM] {tag} alloc_gb min={alloc_min:.2f} max={alloc_max:.2f} mean={alloc_mean:.2f}")
        print(f"[MEM] {tag} reserv_gb min={res_min:.2f} max={res_max:.2f} mean={res_mean:.2f}")


def main():
    args = parse_args()
    deepspeed.init_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    tp_group, dp_group, tp_rank, dp_rank = build_tp_dp_groups(
        rank, world_size, args.tp_size, args.dp_size
    )

    dtype = get_precision_dtype(args.precision)

    torch.cuda.reset_peak_memory_stats(device)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype)
    model = model.to(device)
    gather_and_print("after_model_load", device, rank, world_size)

    if args.mode == "traditional":
        model = deepspeed.tp_model_init(model, tp_size=args.tp_size, dtype=dtype)

    ds_config = {
        "train_batch_size": args.batch_size * args.dp_size,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {"stage": args.zero_stage},
        "tensor_parallel": {"autotp_size": args.tp_size},
        "data_parallel_size": args.dp_size,
    }
    if args.precision == "bf16":
        ds_config["bf16"] = {"enabled": True}
    elif args.precision == "fp16":
        ds_config["fp16"] = {"enabled": True}

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    mpu = ModelParallelUnit(tp_group, dp_group, args.tp_size, args.dp_size, tp_rank, dp_rank)

    torch.cuda.reset_peak_memory_stats(device)
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_config, mpu=mpu)
    gather_and_print("after_initialize", device, rank, world_size)

    vocab_size = model.config.vocab_size
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.num_steps):
        if tp_rank == 0:
            input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_length), device=device)
            labels = input_ids.clone()
        else:
            input_ids = torch.empty((args.batch_size, args.seq_length), dtype=torch.long, device=device)
            labels = torch.empty((args.batch_size, args.seq_length), dtype=torch.long, device=device)

        tp_src_rank = dp_rank * args.tp_size
        broadcast_inputs(input_ids, labels, tp_group, tp_src_rank)
        outputs = engine(input_ids=input_ids, labels=labels)
        engine.backward(outputs.loss)
        engine.step()

    gather_and_print("after_train", device, rank, world_size)

    if rank == 0:
        print(f"AutoTP memory compare completed for mode={args.mode}.")


if __name__ == "__main__":
    main()
