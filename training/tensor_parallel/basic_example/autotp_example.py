import argparse
from dataclasses import dataclass

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
    parser = argparse.ArgumentParser(description="AutoTP training example (distilled from verify_autotp).")
    parser.add_argument("--local_rank", type=int, default=-1, help="Passed by deepspeed/torchrun.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--tp_size", type=int, default=4)
    parser.add_argument("--dp_size", type=int, default=2)
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
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


def main():
    args = parse_args()
    deepspeed.init_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    tp_group, dp_group, tp_rank, dp_rank = build_tp_dp_groups(
        rank, world_size, args.tp_size, args.dp_size
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = model.to(device)

    # AutoTP is enabled via the DeepSpeed config.
    ds_config = {
        "train_batch_size": args.batch_size * args.dp_size,
        "train_micro_batch_size_per_gpu": args.batch_size,
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
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_config, mpu=mpu)

    vocab_size = model.config.vocab_size
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

    if rank == 0:
        print("AutoTP example completed.")


if __name__ == "__main__":
    main()
