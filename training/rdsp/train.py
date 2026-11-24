# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Train a Hugging Face causal LM with rdsp.

    python train.py --model Qwen/Qwen3-0.6B --stages 4 --microbatches 8
    python train.py --stages 3 --cuts 9,18 \
        --stage 0:gpus=2,zero=2 --stage 1:gpus=2,tp=2 --stage 2:gpus=2

Requires an attached or local Ray cluster with enough GPUs for the layout.
"""

import argparse
import itertools
import json
import time

import ray
import torch
import torch.nn.functional as F
import transformers

import ray_deepspeed_pipeline as rdsp
from ray_deepspeed_pipeline.config import StageOverride

_STAGE_KEYS = {
    "gpus": "num_gpus",
    "zero": "zero_stage",
    "tp": "tp",
    "sp": "sp",
    "ep": "ep",
    "recompute_blocks": "recompute_blocks",
}


def parse_stage(spec: str) -> StageOverride:
    """'1:gpus=2,tp=2' -> StageOverride(stage=1, num_gpus=2, tp=2). Keys: gpus,
    zero, tp, sp, ep, fold, recompute, offload, compile, compile_vision,
    recompute_blocks."""
    index, _, fields = spec.partition(":")
    kwargs = {"stage": int(index)}
    for field in filter(None, fields.split(",")):
        key, _, value = field.partition("=")
        if key in ("fold", "recompute", "offload", "compile", "compile_vision"):
            name = "offload_optimizer" if key == "offload" else key
            kwargs[name] = value.lower() in ("1", "true", "yes")
        elif key in _STAGE_KEYS:
            kwargs[_STAGE_KEYS[key]] = int(value)
        else:
            raise argparse.ArgumentTypeError(f"unknown stage key {key!r} in {spec!r}")
    return StageOverride(**kwargs)


def load_model(name: str, dtype: torch.dtype):
    model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    if model.get_output_embeddings().weight is model.get_input_embeddings().weight:
        # rdsp rejects parameters shared across stages
        head = model.get_output_embeddings()
        head.weight = torch.nn.Parameter(model.get_input_embeddings().weight.detach().clone())
        model.config.tie_word_embeddings = False
    return model


def token_stream(args, tokenizer):
    """Endless stream of token ids: WikiText-103 or uniform random tokens."""
    if args.data == "synthetic":
        gen = torch.Generator().manual_seed(args.seed)
        while True:
            yield from torch.randint(0, len(tokenizer), (65536,), generator=gen).tolist()
    import datasets

    split = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    while True:
        for start in range(0, len(split), 1000):
            text = "".join(split[start : start + 1000]["text"])
            yield from tokenizer(text)["input_ids"]


def microbatches(args, tokenizer):
    """(input_ids, labels) pairs of shape [rows, seq]; labels are the inputs
    shifted by one token."""
    tokens = token_stream(args, tokenizer)
    size = args.rows * (args.seq + 1)
    while True:
        block = torch.tensor(list(itertools.islice(tokens, size))).view(args.rows, args.seq + 1)
        yield block[:, :-1].contiguous(), block[:, 1:].contiguous()


def loss_fn(logits, labels):
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))


def ds_config(args) -> dict:
    if args.deepspeed_config:
        with open(args.deepspeed_config) as f:
            return json.load(f)
    return {
        "train_batch_size": args.rows * args.microbatches,
        "gradient_accumulation_steps": args.microbatches,
        # torch_adam: PyTorch's AdamW, not DeepSpeed's JIT-compiled CUDA op.
        # Optimizer offload needs DeepSpeed's CPU Adam (compiled on first use)
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "torch_adam": not any(o.offload_optimizer for o in args.stage),
            },
        },
        "bf16": {"enabled": args.dtype == "bf16"},
        "zero_optimization": {"stage": args.zero},
        "gradient_clipping": 0.0,
        "steps_per_print": 10**9,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--stages", type=int, default=2)
    p.add_argument(
        "--cuts",
        default="",
        help="block indices where stages 1.. start, e.g. 9,18,27; "
        "'balanced' for cost-balanced cuts; default: even layer counts",
    )
    p.add_argument(
        "--stage",
        type=parse_stage,
        action="append",
        default=[],
        help=(
            "per-stage layout, e.g. 1:gpus=2,tp=2 (repeatable); recompute_blocks=N "
            "recomputes the stage's first N decoder layers, and a stage holding the "
            "vision encoder always recomputes the encoder"
        ),
    )
    p.add_argument("--microbatches", type=int, default=8)
    p.add_argument("--rows", type=int, default=4, help="sequences per microbatch")
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--zero", type=int, default=0, help="default ZeRO stage for every stage")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--data", choices=("wikitext", "synthetic"), default="wikitext")
    p.add_argument("--deepspeed_config", default="", help="JSON file; overrides the flags above")
    p.add_argument("--save_dir", default="", help="checkpoint here after the last step")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    torch.manual_seed(args.seed)
    ray.init(ignore_reinit_error=True)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = load_model(args.model, dtype)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    if args.cuts == "balanced":
        partition = rdsp.BalancedTransformerBlocks()
    elif args.cuts:
        partition = rdsp.ExplicitCuts(tuple(int(c) for c in args.cuts.split(",")))
    else:
        partition = rdsp.UniformTransformerBlocks()
    engine, _, _, _ = rdsp.initialize(
        model=model,
        config=ds_config(args),
        loss_fn=loss_fn,
        pipeline_config=rdsp.PipelineConfig(
            stages=args.stages, partition=partition, stage_overrides=tuple(args.stage)
        ),
    )
    del model  # the stages hold their own copies

    data = microbatches(args, tokenizer)
    tokens_per_step = args.microbatches * args.rows * args.seq
    for step in range(args.steps):
        start = time.perf_counter()
        loss = float(engine.train_batch(data_iter=data))
        elapsed = time.perf_counter() - start
        print(
            f"step {step:5d}  loss {loss:.4f}  {elapsed * 1000:7.1f} ms  "
            f"{tokens_per_step / elapsed / 1000:7.1f}k tok/s",
            flush=True,
        )
    if args.save_dir:
        engine.save_checkpoint(args.save_dir)
        print(f"checkpoint: {args.save_dir}/global_step{engine.global_steps}")
    return engine


if __name__ == "__main__":
    main()
