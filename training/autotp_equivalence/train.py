# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Train one model for N steps and record the loss of every step.

The point of this script is to be run twice at different AutoTP sizes. Tensor
parallelism partitions a layer's arithmetic across ranks; it is not supposed to
change what the layer computes. So two runs that differ only in ``autotp_size``
must follow the same loss curve, and ``compare_loss.py`` checks that they do.

Everything that could make two runs diverge for reasons unrelated to sharding is
pinned here:

* the batches are drawn from a CPU generator seeded identically on every rank,
  so all ranks -- and both runs -- see the same tokens in the same order (AutoTP
  replicates the input across the tensor-parallel group, so ranks must agree);
* weights are fp32, because bf16 rounding noise is far larger than the
  reassociation error this is trying to measure;
* the model must not consume RNG in its forward, or the two runs would diverge
  through dropout rather than through sharding; this is asserted at startup
  rather than assumed.

The loss itself is not interesting -- the model is training on random tokens.
What matters is that two differently-sharded runs agree on it.
"""

import argparse
import json
import os
import time

if os.environ.get("DS_ACCELERATOR") == "cpu":
    # torch's AdamW probes the accelerator's current stream on every step. On a CPU
    # run that still initializes a CUDA context, on a device picked by local rank,
    # which fails outright if that device is busy -- and has nothing to do with the
    # comparison. Hiding the GPUs before torch is imported keeps a CPU run on CPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import deepspeed
import torch
from transformers import AutoConfig, AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--deepspeed_config", required=True)
    parser.add_argument("--metrics_file", required=True, help="JSONL sink, one object per step")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def batch_generator(seed: int) -> torch.Generator:
    """A generator seeded the same way on every rank, so all ranks agree on the data.

    The tensor-parallel group replicates its input, so ranks that disagreed on the
    batch would make the comparison meaningless. Seeding on the CPU keeps the
    stream identical regardless of accelerator.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def assert_no_dropout(config) -> None:
    """Refuse to compare runs whose forward would consume RNG."""
    for name in ("attention_dropout", "hidden_dropout", "resid_pdrop", "embd_pdrop", "dropout"):
        rate = getattr(config, name, 0.0)
        if rate:
            raise ValueError(f"{name}={rate}: dropout would make the two runs diverge through RNG "
                             f"rather than through sharding")


def main() -> None:
    args = parse_args()
    deepspeed.init_distributed()

    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    assert_no_dropout(config)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, dtype=torch.float32)

    with open(args.deepspeed_config) as f:
        ds_config = json.load(f)
    ds_config["train_micro_batch_size_per_gpu"] = args.micro_batch_size
    ds_config["gradient_accumulation_steps"] = 1

    autotp_size = ds_config.get("tensor_parallel", {}).get("autotp_size", 1)
    world_size = torch.distributed.get_world_size()
    if world_size != autotp_size:
        # Spare ranks would become a data-parallel dimension, which averages gradients
        # over more samples and changes what is being compared -- silently.
        raise ValueError(f"world_size ({world_size}) must equal autotp_size ({autotp_size}); "
                         f"this comparison is only meaningful without data parallelism")

    engine, *_ = deepspeed.initialize(model=model, config=ds_config, model_parameters=model.parameters())

    generator = batch_generator(args.seed)
    device = engine.device
    rank = torch.distributed.get_rank()

    if rank == 0:
        directory = os.path.dirname(args.metrics_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        sink = open(args.metrics_file, "w")
        print(f"[autotp_eq] model={args.model_name_or_path} autotp_size={autotp_size} "
              f"world_size={world_size} steps={args.steps} seq_len={args.seq_len} dtype=float32")
    else:
        sink = None

    started = time.time()
    for step in range(args.steps):
        input_ids = torch.randint(0,
                                  config.vocab_size, (args.micro_batch_size, args.seq_len),
                                  generator=generator,
                                  dtype=torch.long).to(device)

        loss = engine(input_ids=input_ids, labels=input_ids).loss
        engine.backward(loss)
        engine.step()

        if rank == 0:
            sink.write(json.dumps({"step": step, "loss": loss.item(), "autotp_size": autotp_size}) + "\n")
            sink.flush()
            if step % args.log_every == 0 or step == args.steps - 1:
                print(f"[autotp_eq][step {step}] loss={loss.item():.6f} "
                      f"elapsed={time.time() - started:.1f}s",
                      flush=True)

    if sink is not None:
        sink.close()
        print(f"[autotp_eq] wrote {args.steps} steps to {args.metrics_file} "
              f"in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
