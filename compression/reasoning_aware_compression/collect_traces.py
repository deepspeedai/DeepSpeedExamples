# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Phase I of RAC: collect on-policy chain-of-thought traces from a dense model.

The traces are the model's *own* rollouts on task prompts, which is what makes
the calibration activations match what the model computes while decoding.
Output is a JSONL file consumed by ``prune.py --calibration rac``.

Three generation backends:

``vllm``
    Fastest single-node option; use it whenever vLLM is installed.
``hf``
    Plain ``transformers.generate``; works anywhere, slow for 8k-token traces.
``deepspeed``
    ZeRO-Inference: ZeRO-3 shards the weights across ranks and can offload them
    to CPU or NVMe, so a 32B/70B dense model generates its traces on a small
    number of GPUs. Launch with the DeepSpeed launcher, e.g.
    ``deepspeed --num_gpus 2 collect_traces.py --backend deepspeed ...``.

Example::

    python collect_traces.py \\
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \\
        --dataset open-r1/OpenR1-Math-220k --prompt-column problem \\
        --output traces/math_1.5b.jsonl --trace-tokens 1_000_000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import List

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
    "You first think about the reasoning process as an internal monologue and then provide "
    "the user with the answer.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    model = parser.add_argument_group("model")
    model.add_argument("--model", required=True, help="Dense reasoning model to roll out.")
    model.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    model.add_argument("--trust-remote-code", action="store_true")

    data = parser.add_argument_group("prompts")
    data.add_argument("--dataset",
                      required=True,
                      help="HF dataset name or path to a local .jsonl file of prompts.")
    data.add_argument("--dataset-config", default=None)
    data.add_argument("--dataset-split", default="train")
    data.add_argument("--prompt-column", default="problem",
                      help="'problem' for OpenR1-Math-220k, 'prompt' for Codeforces.")
    data.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    data.add_argument("--no-chat-template", action="store_true",
                      help="Feed raw prompts instead of applying the chat template.")
    data.add_argument("--max-prompts", type=int, default=None,
                      help="Cap on prompts to roll out (the token budget usually binds first).")

    gen = parser.add_argument_group("generation")
    gen.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf", "deepspeed"])
    gen.add_argument("--trace-tokens", type=int, default=1_000_000,
                     help="Stop once this many prompt+completion tokens have been collected. "
                          "The paper uses 1M.")
    gen.add_argument("--max-new-tokens", type=int, default=8192,
                     help="T_max in Algorithm 1 of the paper.")
    gen.add_argument("--num-generations", type=int, default=1,
                     help="On-policy rollouts per prompt.")
    gen.add_argument("--temperature", type=float, default=0.6)
    gen.add_argument("--top-p", type=float, default=0.95)
    gen.add_argument("--batch-size", type=int, default=8, help="Prompts per generation call.")
    gen.add_argument("--seed", type=int, default=0)

    ds = parser.add_argument_group("deepspeed backend")
    ds.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallelism.")
    ds.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="vLLM only.")
    ds.add_argument("--cpu-offload", action="store_true",
                    help="ZeRO-Inference: offload parameters to CPU.")
    ds.add_argument("--nvme-offload-dir", default=None,
                    help="ZeRO-Inference: offload parameters to this NVMe directory.")
    ds.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)),
                    help="Set by the DeepSpeed launcher.")

    parser.add_argument("--output", required=True, help="Destination .jsonl file.")
    return parser.parse_args()


def load_prompts(args, tokenizer) -> List[str]:
    """Chat-templated prompts, ready to be fed to the model."""
    from rac.calibration import chat_template_prompt

    if args.dataset.endswith(".jsonl") or args.dataset.endswith(".json"):
        raw = []
        with open(args.dataset) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    raw.append(json.loads(line)[args.prompt_column])
    else:
        from datasets import load_dataset

        dataset = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split)
        if args.prompt_column not in dataset.column_names:
            raise KeyError(f"Column '{args.prompt_column}' not in {args.dataset} "
                           f"(available: {dataset.column_names}). Set --prompt-column.")
        raw = list(dataset[args.prompt_column])

    if args.max_prompts is not None:
        raw = raw[:args.max_prompts]

    return [
        chat_template_prompt(tokenizer, p, args.system_prompt, not args.no_chat_template)
        for p in raw if p
    ]


def generate_vllm(args, prompts: List[str], writer) -> int:
    """Roll out with vLLM, in chunks, until the token budget is exhausted."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
    )
    sampling = SamplingParams(
        n=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    total = 0
    for start in range(0, len(prompts), args.batch_size):
        chunk = prompts[start:start + args.batch_size]
        for output in llm.generate(chunk, sampling):
            prompt_tokens = len(output.prompt_token_ids)
            for candidate in output.outputs:
                total = writer(output.prompt, candidate.text, prompt_tokens,
                               len(candidate.token_ids))
        if total >= args.trace_tokens:
            break
    return total


def build_hf_model(args):
    """Plain ``transformers`` model on a single device."""
    import torch
    from transformers import AutoModelForCausalLM

    from rac import dtype_kwargs

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        **dtype_kwargs(getattr(torch, args.dtype)),
    )
    model.eval()
    return model.to("cuda" if torch.cuda.is_available() else "cpu")


def build_deepspeed_model(args):
    """ZeRO-Inference: ZeRO-3 sharding with optional CPU/NVMe parameter offload.

    Follows ``inference/huggingface/zero_inference/run_model.py`` in this repo.
    All ranks execute the same ``generate`` call with ``synced_gpus=True``;
    only rank 0 writes the traces.
    """
    import deepspeed
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.integrations.deepspeed import HfDeepSpeedConfig

    deepspeed.init_distributed()
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    hidden_size = config.hidden_size
    dtype = getattr(torch, args.dtype)

    ds_config = {
        "fp16": {"enabled": dtype == torch.float16},
        "bf16": {"enabled": dtype == torch.bfloat16},
        "zero_optimization": {
            "stage": 3,
            "stage3_prefetch_bucket_size": 2 * hidden_size * hidden_size,
            "stage3_param_persistence_threshold": hidden_size,
            "stage3_max_live_parameters": 2 * hidden_size * hidden_size,
        },
        "train_batch_size": args.batch_size * int(os.environ.get("WORLD_SIZE", 1)),
        "steps_per_print": 2000,
        "wall_clock_breakdown": False,
    }

    if args.nvme_offload_dir:
        Path(args.nvme_offload_dir).mkdir(parents=True, exist_ok=True)
        ds_config["zero_optimization"]["offload_param"] = {
            "device": "nvme",
            "nvme_path": args.nvme_offload_dir,
            "pin_memory": True,
            "buffer_count": 5,
            "buffer_size": 2 * (1024**3),
        }
        ds_config["aio"] = {
            "block_size": 1048576 * 16,
            "queue_depth": 64,
            "thread_count": 8,
            "single_submit": False,
            "overlap_events": True,
        }
    elif args.cpu_offload:
        ds_config["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": True}

    # Instructs from_pretrained to materialise the weights directly as ZeRO-3
    # shards; must stay alive until the model is built.
    dschf = HfDeepSpeedConfig(ds_config)  # noqa: F841

    from rac import dtype_kwargs

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        **dtype_kwargs(dtype),
    )
    model.eval()

    engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    engine.module.eval()
    return engine.module


def _distributed_world() -> int:
    """Number of ranks participating, or 1 when running unsharded."""
    try:
        import torch.distributed as dist
    except ImportError:
        return 1
    if not (dist.is_available() and dist.is_initialized()):
        return 1
    return dist.get_world_size()


def assert_ranks_agree(generated, device) -> None:
    """Check that every rank decoded the same tokens.

    ZeRO-Inference shards parameters, not data: each rank runs the same batch
    and must produce identical sequences, which is what makes it safe for rank 0
    alone to write the traces. Under a different sharding scheme (or a
    rank-dependent seed) the ranks would diverge, and rank 0's file would
    describe rollouts the other ranks never made -- so compare a cheap checksum
    against rank 0 and fail loudly instead of writing inconsistent traces.
    """
    import torch
    import torch.distributed as dist

    if _distributed_world() < 2:
        return

    checksum = torch.tensor(
        [generated.shape[0], generated.shape[1],
         int(generated.sum().item())],
        dtype=torch.int64,
        device=device,
    )
    reference = checksum.clone()
    dist.broadcast(reference, src=0)
    if not torch.equal(checksum, reference):
        raise RuntimeError(
            f"Rank {dist.get_rank()} generated different tokens from rank 0 "
            f"(shape/checksum {checksum.tolist()} vs {reference.tolist()}). Trace collection "
            "assumes ZeRO-3 parameter sharding with replicated data, so every rank decodes the "
            "same sequences; only rank 0 writes them.")


def _max_total_across_ranks(total: int, device) -> int:
    """Largest running token count over all ranks.

    Every rank has to leave the generation loop on the same iteration: with
    ``synced_gpus=True`` a rank that keeps going waits forever for peers that
    stopped. The per-rank counts are expected to be equal -- see
    :func:`assert_ranks_agree` -- but reducing them makes the exit collective
    rather than a per-rank assumption.
    """
    import torch
    import torch.distributed as dist

    if _distributed_world() < 2:
        return total

    counter = torch.tensor([total], dtype=torch.int64, device=device)
    dist.all_reduce(counter, op=dist.ReduceOp.MAX)
    return int(counter.item())


def generate_transformers(args, prompts: List[str], writer, use_deepspeed: bool) -> int:
    """Roll out with ``model.generate`` (HF or ZeRO-Inference backend)."""
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=args.trust_remote_code,
                                              padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_deepspeed_model(args) if use_deepspeed else build_hf_model(args)
    device = next(model.parameters()).device if not use_deepspeed else torch.device(
        "cuda", args.local_rank)
    is_writer_rank = int(os.environ.get("RANK", 0)) == 0

    # Every rank counts tokens, but only rank 0 writes: under ZeRO-Inference all
    # ranks must run the same number of generate() calls or they deadlock, so
    # the stopping condition cannot depend on who is writing.
    total = 0
    torch.manual_seed(args.seed)
    for start in range(0, len(prompts), args.batch_size):
        chunk = prompts[start:start + args.batch_size]
        encoded = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
        # Some tokenizers also return token_type_ids, which causal LMs reject.
        batch = {
            k: v.to(device)
            for k, v in encoded.items() if k in ("input_ids", "attention_mask")
        }

        with torch.no_grad():
            generated = model.generate(
                **batch,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.num_generations,
                pad_token_id=tokenizer.pad_token_id,
                synced_gpus=use_deepspeed,
            )

        if use_deepspeed:
            assert_ranks_agree(generated, device)

        prompt_len = batch["input_ids"].shape[1]
        completions = generated[:, prompt_len:]
        for i, sequence in enumerate(completions):
            prompt = chunk[i // args.num_generations]
            text = tokenizer.decode(sequence, skip_special_tokens=True)
            # Number of *kept* tokens, not the generated length: generate() right-pads
            # every sequence in the batch out to the longest one, and pad_token was
            # aliased to eos_token above, so the terminal eos is not counted either.
            # That matches the text actually written to the trace file, which is what
            # the token budget is meant to track.
            n_completion = int((sequence != tokenizer.pad_token_id).sum())
            n_prompt = int(batch["attention_mask"][i // args.num_generations].sum())
            total += n_prompt + n_completion
            if is_writer_rank:
                writer(prompt, text, n_prompt, n_completion)

        if _max_total_across_ranks(total, device) >= args.trace_tokens:
            break
    return total


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=args.trust_remote_code)
    prompts = load_prompts(args, tokenizer)
    print(f"[traces] {len(prompts)} prompts loaded from {args.dataset}")

    backend = args.backend
    if backend == "auto":
        backend = "vllm" if importlib.util.find_spec("vllm") else "hf"
        print(f"[traces] backend auto-selected: {backend}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    written = {"traces": 0, "tokens": 0}
    started = time.time()

    with output.open("w") as handle:

        def writer(prompt: str, completion: str, n_prompt: int, n_completion: int) -> int:
            """Append one trace; returns the running token total."""
            handle.write(
                json.dumps({
                    "prompt": prompt,
                    "completion": completion,
                    "prompt_tokens": n_prompt,
                    "completion_tokens": n_completion,
                    "model": args.model,
                    "dataset": args.dataset,
                }) + "\n")
            handle.flush()
            written["traces"] += 1
            written["tokens"] += n_prompt + n_completion
            if written["traces"] % 10 == 0:
                print(f"[traces] {written['traces']} traces, "
                      f"{written['tokens']:,}/{args.trace_tokens:,} tokens, "
                      f"{time.time() - started:.0f}s")
            return written["tokens"]

        if backend == "vllm":
            generate_vllm(args, prompts, writer)
        else:
            generate_transformers(args, prompts, writer, use_deepspeed=backend == "deepspeed")

    print(f"[traces] wrote {written['traces']} traces "
          f"({written['tokens']:,} tokens) to {output}")
    if written["tokens"] < args.trace_tokens:
        print(f"[traces] WARNING: token budget {args.trace_tokens:,} not reached; "
              "raise --max-prompts or use a larger prompt dataset.")


if __name__ == "__main__":
    main()
