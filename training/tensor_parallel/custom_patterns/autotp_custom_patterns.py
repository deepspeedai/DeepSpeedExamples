import argparse
from dataclasses import dataclass
from typing import Iterable, List

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

IGNORE_INDEX = -100


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


class ToyTextDataset(Dataset):
    def __init__(self, tokenizer, seq_length: int):
        texts = [
            "DeepSpeed makes distributed training faster.",
            "AutoTP shards large layers across GPUs.",
            "Tensor parallelism reduces per-GPU memory.",
            "ZeRO optimizes optimizer state memory.",
            "This is a small in-memory dataset.",
            "We are testing AutoTP training.",
            "Distributed training requires careful data sharding.",
            "Sharded model weights reduce memory pressure.",
            "This example uses a custom AutoTP config.",
            "Random samplers ensure data diversity.",
        ]
        self.samples = []
        for text in texts:
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=seq_length,
                add_special_tokens=True,
            )
            input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            self.samples.append((input_ids, attention_mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, attention_mask = self.samples[idx]
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class DPRandomSampler(Sampler[int]):
    """Random sampler sharded by DP rank."""

    def __init__(self, data_source: Dataset, dp_rank: int, dp_size: int, seed: int):
        self.data_source = data_source
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterable[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_source), generator=g).tolist()
        return iter(indices[self.dp_rank :: self.dp_size])

    def __len__(self) -> int:
        return (len(self.data_source) + self.dp_size - 1) // self.dp_size


def collate_batch(samples: List[dict], pad_token_id: int) -> dict:
    input_ids = [s["input_ids"] for s in samples]
    labels = [s["labels"] for s in samples]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=IGNORE_INDEX
    )
    attention_mask = input_ids.ne(pad_token_id)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def parse_args():
    parser = argparse.ArgumentParser(description="AutoTP custom patterns example.")
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-4-multimodal-instruct")
    parser.add_argument("--tp_size", type=int, default=4)
    parser.add_argument("--dp_size", type=int, default=2)
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_tp_dp_groups(rank: int, world_size: int, tp_size: int, dp_size: int):
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


def main():
    args = parse_args()
    deepspeed.init_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    tp_group, dp_group, tp_rank, dp_rank = build_tp_dp_groups(
        rank, world_size, args.tp_size, args.dp_size
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    # AutoTP is enabled via the DeepSpeed config.
    ds_config = {
        "train_batch_size": args.batch_size * args.dp_size,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "zero_optimization": {"stage": args.zero_stage},
        "tensor_parallel": {
            "autotp_size": args.tp_size,
            "partition_config": {
                "use_default_specs": False,
                "layer_specs": [
                    {
                        "patterns": [".*\\.self_attn\\.qkv_proj\\.weight$"],
                        "partition_type": "column",
                        "shape": (3, -1),
                        "partition_dim": 0,
                    },
                    {
                        "patterns": [".*\\.self_attn\\.o_proj\\.weight$"],
                        "partition_type": "row",
                    },
                    {
                        "patterns": [".*\\.mlp\\.gate_up_proj\\.weight$"],
                        "partition_type": "column",
                        "shape": (2, -1),
                        "partition_dim": 0,
                    },
                    {
                        "patterns": [".*\\.mlp\\.down_proj\\.weight$"],
                        "partition_type": "row",
                    },
                ],
            },
        },
        "data_parallel_size": args.dp_size,
    }
    if args.precision == "bf16":
        ds_config["bf16"] = {"enabled": True}
    elif args.precision == "fp16":
        ds_config["fp16"] = {"enabled": True}

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    mpu = ModelParallelUnit(tp_group, dp_group, args.tp_size, args.dp_size, tp_rank, dp_rank)
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_config, mpu=mpu)

    dataset = ToyTextDataset(tokenizer, args.seq_length)
    sampler = DPRandomSampler(dataset, dp_rank=dp_rank, dp_size=args.dp_size, seed=args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=lambda samples: collate_batch(samples, tokenizer.pad_token_id),
    )

    engine.train()
    data_iter = iter(dataloader)
    for step in range(args.num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            sampler.set_epoch(step)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        outputs = engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        engine.backward(outputs.loss)
        engine.step()

        if rank == 0 and step % 5 == 0:
            print(f"step={step} loss={outputs.loss.item():.4f}")

    if rank == 0:
        print("AutoTP custom patterns example completed.")


if __name__ == "__main__":
    main()
