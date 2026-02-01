#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

"""
Training script demonstrating bf16_master_weights_and_grads and bf16_optimizer_states
options in DeepSpeed for reduced memory usage.

Usage:
    deepspeed --num_gpus=1 train.py --deepspeed_config configs/baseline.json
    deepspeed --num_gpus=1 train.py --deepspeed_config configs/bf16_master_wg.json
    deepspeed --num_gpus=1 train.py --deepspeed_config configs/bf16_full.json
"""

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import deepspeed
from deepspeed.accelerator import get_accelerator
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class SimpleTransformerBlock(nn.Module):
    """A simple transformer block for demonstration."""

    def __init__(self, hidden_dim, num_heads=8, ff_dim=None):
        super().__init__()
        ff_dim = ff_dim or hidden_dim * 4

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
        )

    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class SimpleTransformerModel(nn.Module):
    """A simple transformer model for memory benchmarking."""

    def __init__(self, vocab_size=50000, hidden_dim=1024, num_layers=12, num_heads=16, max_seq_len=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.activation_checkpointing = False

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        self.layers = nn.ModuleList([
            SimpleTransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def enable_activation_checkpointing(self):
        self.activation_checkpointing = True

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        x = self.embedding(input_ids) + self.pos_embedding(positions)

        for layer in self.layers:
            if self.activation_checkpointing:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        logits = self.output_proj(x)
        return logits


def get_args():
    parser = argparse.ArgumentParser(description="BF16 Low-Precision Master Weights Demo")

    # Model configuration
    parser.add_argument("--hidden_dim", type=int, default=1024, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--vocab_size", type=int, default=50000, help="Vocabulary size")

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length")
    parser.add_argument("--num_steps", type=int, default=20, help="Number of training steps")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warmup steps before measuring")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")

    # DeepSpeed configuration
    parser.add_argument("--deepspeed_config", type=str, required=True, help="Path to DeepSpeed config")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    # Logging
    parser.add_argument("--log_interval", type=int, default=5, help="Log interval")

    # Activation checkpointing
    parser.add_argument("--activation_checkpointing", action="store_true", help="Enable activation checkpointing")

    # Loss logging
    parser.add_argument("--loss_log_file", type=str, default=None, help="File to save loss values for plotting")

    # Dataset
    parser.add_argument("--use_real_data", action="store_true", help="Use wikitext dataset instead of random data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()


def load_wikitext_data(tokenizer_name, seq_length, batch_size, world_size, rank):
    """Load wikitext dataset for training."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"[Rank {rank}] Loading wikitext dataset...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train[:5%]')

    # Filter empty texts
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 50)

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            max_length=seq_length,
            truncation=True,
            return_tensors=None
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # Create distributed sampler and dataloader
    sampler = DistributedSampler(tokenized_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)

    print(f"[Rank {rank}] Dataset loaded: {len(tokenized_dataset)} examples, vocab_size={tokenizer.vocab_size}")
    return dataloader, tokenizer.vocab_size


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_memory_stats():
    """Get current GPU memory statistics."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "peak": 0}

    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak": torch.cuda.max_memory_allocated(),
    }


def format_memory(bytes_val):
    """Format bytes to human readable string."""
    gb = bytes_val / (1024 ** 3)
    return f"{gb:.2f} GB"


def main():
    args = get_args()

    # Initialize distributed
    deepspeed.init_distributed()
    local_rank = args.local_rank
    if local_rank == -1:
        local_rank = int(deepspeed.comm.get_rank())

    world_size = deepspeed.comm.get_world_size()

    device = get_accelerator().device_name(local_rank)
    torch.cuda.set_device(local_rank)

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    # Load real data if requested
    dataloader = None
    actual_vocab_size = args.vocab_size
    if args.use_real_data:
        dataloader, actual_vocab_size = load_wikitext_data(
            tokenizer_name="gpt2",
            seq_length=args.seq_length,
            batch_size=args.batch_size,
            world_size=world_size,
            rank=local_rank
        )

    # Create model
    print(f"[Rank {local_rank}] Creating model with hidden_dim={args.hidden_dim}, "
          f"num_layers={args.num_layers}, num_heads={args.num_heads}, vocab_size={actual_vocab_size}")

    model = SimpleTransformerModel(
        vocab_size=actual_vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_length,
    )

    total_params, trainable_params = count_parameters(model)
    print(f"[Rank {local_rank}] Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Enable activation checkpointing if requested
    if args.activation_checkpointing:
        model.enable_activation_checkpointing()
        print(f"[Rank {local_rank}] Activation checkpointing enabled")

    # Read config to check if torch_autocast is enabled
    import json
    with open(args.deepspeed_config, 'r') as f:
        ds_config = json.load(f)

    use_autocast = ds_config.get("torch_autocast", {}).get("enabled", False)
    autocast_dtype_str = ds_config.get("torch_autocast", {}).get("dtype", "torch.bfloat16")
    autocast_dtype = torch.bfloat16 if "bfloat16" in autocast_dtype_str else torch.float16

    # Initialize DeepSpeed - use config file path directly (not via args to avoid conflict)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=args.deepspeed_config,
    )

    print(f"[Rank {local_rank}] DeepSpeed initialized with config: {args.deepspeed_config}")

    mem_after_init = get_memory_stats()
    print(f"[Rank {local_rank}] Memory after init: allocated={format_memory(mem_after_init['allocated'])}, "
          f"reserved={format_memory(mem_after_init['reserved'])}")

    # Training loop
    model_engine.train()
    loss_fn = nn.CrossEntropyLoss()

    total_time = 0
    step_times = []
    loss_history = []

    # Setup data iterator
    if dataloader is not None:
        data_iter = iter(dataloader)

    for step in range(args.num_steps):
        start_time = time.time()

        # Get input data
        if dataloader is not None:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            input_ids = batch['input_ids'].to(device)
            labels = input_ids.clone()  # For language modeling, labels = input_ids
        else:
            # Generate random input data
            input_ids = torch.randint(0, actual_vocab_size, (args.batch_size, args.seq_length), device=device)
            labels = torch.randint(0, actual_vocab_size, (args.batch_size, args.seq_length), device=device)

        # Forward pass with optional autocast
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits = model_engine(input_ids)
                loss = loss_fn(logits.view(-1, actual_vocab_size), labels.view(-1))
        else:
            logits = model_engine(input_ids)
            loss = loss_fn(logits.view(-1, actual_vocab_size), labels.view(-1))

        # Backward pass - use PyTorch-style backward
        loss.backward()

        # Optimizer step
        model_engine.step()

        step_time = time.time() - start_time

        if step >= args.warmup_steps:
            step_times.append(step_time)

        # Record loss for plotting
        loss_history.append((step, loss.item()))

        if step % args.log_interval == 0 or step == args.num_steps - 1:
            mem_stats = get_memory_stats()
            print(f"[Rank {local_rank}] Step {step}: loss={loss.item():.4f}, "
                  f"time={step_time:.3f}s, "
                  f"alloc_mem={format_memory(mem_stats['allocated'])}, "
                  f"peak_mem={format_memory(mem_stats['peak'])}")

    # Final statistics
    final_mem = get_memory_stats()
    avg_step_time = sum(step_times) / len(step_times) if step_times else 0

    print("\n" + "=" * 60)
    print(f"[Rank {local_rank}] FINAL RESULTS")
    print(f"  Config: {args.deepspeed_config}")
    print(f"  Model: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}")
    print(f"  Parameters: {total_params:,}")
    print(f"  Batch size: {args.batch_size}, Seq length: {args.seq_length}")
    print(f"  Average step time: {avg_step_time:.3f}s")
    print(f"  Peak memory: {format_memory(final_mem['peak'])}")
    print(f"  Final allocated memory: {format_memory(final_mem['allocated'])}")
    print("=" * 60)

    # Output machine-readable summary line for parsing
    print(f"SUMMARY: config={args.deepspeed_config} params={total_params} "
          f"peak_mem_bytes={final_mem['peak']} alloc_mem_bytes={final_mem['allocated']} "
          f"avg_step_time={avg_step_time:.4f}")

    # Save loss history to file if requested (only rank 0)
    if args.loss_log_file and local_rank == 0:
        import csv
        with open(args.loss_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'loss'])
            writer.writerows(loss_history)
        print(f"Loss history saved to: {args.loss_log_file}")

    model_engine.destroy()


if __name__ == "__main__":
    main()
