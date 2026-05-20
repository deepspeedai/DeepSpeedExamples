import torch
import time
import deepspeed
import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from transformers.integrations.deepspeed import HfDeepSpeedConfig
import json
import random
import numpy as np
from deepspeed import comm as dist
import logging

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import wandb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DATASET_REGISTRY = {
    "sahil2801/CodeAlpaca-20k": {
        "split": "train",
        "preprocessor": "alpaca",
        "field_map": {
            "instruction": "instruction",
            "input": "input",
            "output": "output",
        },
    },
    "meta-math/MetaMathQA": {
        "split": "train",
        "preprocessor": "alpaca",
        "field_map": {
            "instruction": "query",
            "input": None,
            "output": "response",
        },
        "sample_rate": 0.1,
    },
    "cais/mmlu": {
        "subset": "all",
        "split": "auxiliary_train",
        "preprocessor": "mmlu",
        "field_map": None,
    },
}


def load_and_prepare_dataset(dataset_name):
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{dataset_name}' not in DATASET_REGISTRY. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    config = DATASET_REGISTRY[dataset_name]

    load_kwargs = {"path": dataset_name}
    if config.get("subset"):
        load_kwargs["name"] = config["subset"]
    dataset = load_dataset(**load_kwargs)
    raw_dataset = dataset[config["split"]]

    sample_rate = config.get("sample_rate")
    if sample_rate is not None and sample_rate < 1.0:
        n = len(raw_dataset)
        keep = max(1, int(n * sample_rate))
        raw_dataset = raw_dataset.shuffle(seed=42).select(range(keep))
        print(f"Downsampled {dataset_name}: {n} -> {keep} (rate={sample_rate})")

    field_map = config.get("field_map")
    preprocessor = config["preprocessor"]

    if preprocessor == "alpaca" and field_map:
        rename_map = {}
        for target_field, source_field in field_map.items():
            if source_field is not None and source_field != target_field:
                rename_map[source_field] = target_field
        if rename_map:
            raw_dataset = raw_dataset.rename_columns(rename_map)

    if preprocessor == "alpaca" and "input" not in raw_dataset.column_names:
        raw_dataset = raw_dataset.add_column("input", [""] * len(raw_dataset))

    return raw_dataset, preprocessor


def preprocess_alpaca(example, tokenizer, max_length=2048):
    # Build instruction part (will be masked from loss)
    instruction = f"### Instruction:\n{example['instruction']}\n\n"
    if example.get("input", ""):
        instruction += f"### Input:\n{example['input']}\n\n"
    instruction += "### Response:\n"
    response = example["output"]

    full_prompt = instruction + response
    tokenized = tokenizer(
        full_prompt, truncation=True, max_length=max_length, padding="max_length"
    )

    # Find instruction length to mask it from loss
    # Use full_prompt tokenization to get accurate instruction boundary after truncation
    instruction_ids = tokenizer(instruction, add_special_tokens=False)["input_ids"]
    instruction_len = len(instruction_ids)

    # Ensure at least one token is unmasked to avoid NaN loss
    # If instruction is longer than max_length, only mask padding tokens
    seq_len = sum(1 for t in tokenized["input_ids"] if t != tokenizer.pad_token_id)
    if instruction_len >= seq_len:
        instruction_len = max(0, seq_len - 1)  # Keep at least the last non-pad token

    # Mask instruction and padding tokens in labels (set to -100, ignored by CrossEntropyLoss)
    labels = tokenized["input_ids"].copy()
    for i in range(len(labels)):
        if i < instruction_len or labels[i] == tokenizer.pad_token_id:
            labels[i] = -100
    tokenized["labels"] = labels
    return tokenized


def preprocess_magicoder(example, tokenizer, max_length=2048):
    # Magicoder uses 'problem' / 'solution' fields
    instruction = f"### Instruction:\n{example['problem']}\n\n### Response:\n"
    response = example["solution"]

    full_prompt = instruction + response
    tokenized = tokenizer(
        full_prompt, truncation=True, max_length=max_length, padding="max_length"
    )

    instruction_ids = tokenizer(instruction, add_special_tokens=False)["input_ids"]
    instruction_len = len(instruction_ids)

    seq_len = sum(1 for t in tokenized["input_ids"] if t != tokenizer.pad_token_id)
    if instruction_len >= seq_len:
        instruction_len = max(0, seq_len - 1)

    labels = tokenized["input_ids"].copy()
    for i in range(len(labels)):
        if i < instruction_len or labels[i] == tokenizer.pad_token_id:
            labels[i] = -100
    tokenized["labels"] = labels
    return tokenized


def preprocess_mmlu(example, tokenizer, max_length=2048):
    choices = example["choices"]
    labels = "ABCDEFGHIJ"
    instruction = f"### Instruction:\n{example['question']}\n"
    for i, choice in enumerate(choices):
        instruction += f"{labels[i]}. {choice}\n"
    instruction += "\nAnswer with the letter of the correct choice.\n\n### Response:\n"
    answer_letter = example["answer"]
    if isinstance(answer_letter, int):
        answer_letter = labels[answer_letter]
    response = answer_letter

    full_prompt = instruction + response
    tokenized = tokenizer(
        full_prompt, truncation=True, max_length=max_length, padding="max_length"
    )

    instruction_ids = tokenizer(instruction, add_special_tokens=False)["input_ids"]
    instruction_len = len(instruction_ids)

    seq_len = sum(1 for t in tokenized["input_ids"] if t != tokenizer.pad_token_id)
    if instruction_len >= seq_len:
        instruction_len = max(0, seq_len - 1)

    labels_out = tokenized["input_ids"].copy()
    for i in range(len(labels_out)):
        if i < instruction_len or labels_out[i] == tokenizer.pad_token_id:
            labels_out[i] = -100
    tokenized["labels"] = labels_out
    return tokenized


PREPROCESSORS = {
    "alpaca": preprocess_alpaca,
    "magicoder": preprocess_magicoder,
    "mmlu": preprocess_mmlu,
}


def evaluate(model_engine, eval_dataloader):
    import torch
    from tqdm import tqdm
    from deepspeed import comm as dist

    model_engine.eval()
    torch.cuda.empty_cache()
    losses = []
    rank = dist.get_rank() if dist.is_initialized() else 0

    with torch.no_grad():
        if rank == 0:
            enum = tqdm(eval_dataloader, desc="Evaluating", leave=False)
        else:
            enum = eval_dataloader
        for batch in enum:
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            outputs = model_engine(**batch)
            loss = outputs.loss
            losses.append(loss.item())
            del outputs
    model_engine.train()

    if len(losses) == 0:
        return None
    avg_loss = sum(losses) / len(losses)
    return avg_loss

def print_r(rank, arg):
    if rank == dist.get_rank():
        print(arg)


def _save_weights(model_engine, tokenizer, output_dir, step, keep_last=2):
    """Save model weights for the given step; remove old checkpoints beyond keep_last."""
    import shutil
    rank = dist.get_rank()
    ckpt_dir = os.path.join(output_dir, f"step_{step}")
    rank_dir = os.path.join(ckpt_dir, str(rank))
    os.makedirs(rank_dir, exist_ok=True)
    state_dict = model_engine.module.state_dict()
    if rank == 0:
        torch.save(state_dict, os.path.join(rank_dir, "model_weights.pt"))
        tokenizer.save_pretrained(rank_dir)
    else:
        expert_dict = {k: v for k, v in state_dict.items() if v.ndim == 3}
        torch.save(expert_dict, os.path.join(rank_dir, "model_weights.pt"))
    dist.barrier()
    # Remove old checkpoints beyond keep_last (rank 0 only to avoid races)
    if rank == 0:
        ckpts = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("step_")],
            key=lambda d: int(d.split("_")[1]),
        )
        for old_ckpt in ckpts[:-keep_last]:
            shutil.rmtree(os.path.join(output_dir, old_ckpt), ignore_errors=True)
    dist.barrier()
    print_r(0, f"Saved checkpoint to {ckpt_dir}")


def main(args):
    logging.basicConfig(level=logging.INFO, filename="pytorch_log.txt")
    set_seed(args.seed)

    # override batch size in ds_config
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    ds_config["train_batch_size"] = args.batch_size
    delattr(args, "deepspeed_config")
    # make sure models are properly loaded in zero3
    dschf = HfDeepSpeedConfig(ds_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    """
    # the code below allows you to train only part of the parameters
    # we haven't parameterize this part yet, so uncomment down below and modify the code manually

    # Freeze all parameters except gate parameters BEFORE DeepSpeed initialization
    # This needs to be done before passing to DeepSpeed
    for name, param in model.named_parameters():
        if 'gate' in name.lower() and not 'gate_proj' in name.lower():
            param.requires_grad = True
            print(f"Unfrozen parameter: {name}")
        else:
            param.requires_grad = False

    # Enable input gradient requirements to ensure gradient flow
    # This is needed when using gradient checkpointing with partially frozen models
    model.enable_input_require_grads()
    """

    # Load dataset and split into train/eval
    if args.dataset_name in DATASET_REGISTRY:
        raw_dataset, preprocessor_name = load_and_prepare_dataset(args.dataset_name)
        preprocess_fn = PREPROCESSORS[preprocessor_name]
    else:
        dataset = load_dataset(args.dataset_name)
        raw_dataset = dataset["train"]
        if "problem" in raw_dataset.column_names:
            preprocessor_name = "magicoder"
        else:
            preprocessor_name = "alpaca"
        preprocess_fn = PREPROCESSORS[preprocessor_name]

    split_dataset = raw_dataset.train_test_split(test_size=0.05, seed=args.seed)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    keep_cols = {"input_ids", "attention_mask", "labels"}
    tokenized_train_dataset = train_dataset.map(
        lambda x: preprocess_fn(x, tokenizer, max_length=args.max_length),
        batched=False,
        remove_columns=[c for c in train_dataset.column_names if c not in keep_cols],
    )
    tokenized_eval_dataset = eval_dataset.map(
        lambda x: preprocess_fn(x, tokenizer, max_length=args.max_length),
        batched=False,
        remove_columns=[c for c in eval_dataset.column_names if c not in keep_cols],
    )

    eval_dataloader = DataLoader(
        tokenized_eval_dataset,
        batch_size=args.eval_batch_size,
        collate_fn=default_data_collator,
        shuffle=False,
        drop_last=True,
    )

    model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=tokenized_train_dataset,
        collate_fn=default_data_collator,
        config=ds_config,
    )

    train_sampler = DistributedSampler(
        tokenized_train_dataset,
        shuffle=True,
        seed=args.seed,
    )
    per_device_batch = model_engine.train_micro_batch_size_per_gpu()
    train_dataloader = DataLoader(
        tokenized_train_dataset,
        batch_size=per_device_batch,
        sampler=train_sampler,
        collate_fn=default_data_collator,
        drop_last=True,
    )

    model_engine.train()
    global_step = 0
    total_time = 0
    total_count = 0

    # skip unnecessary evaluation and checkpoint saving
    save_checkpoint_p = True
    if args.bench_start >= 0 and args.bench_steps > 0:
        save_checkpoint_p = False
    if args.profile_start >= 0:
        save_checkpoint_p = False

    if args.profile_start >= 0:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
        )
    else:
        prof = None

    # setup logging
    if args.wandb_name != None and dist.get_rank() == 0:
        wandb.init(project="deepspeed_finetune_demo", name=args.wandb_name)

    global_samples = 0
    for epoch in range(args.num_train_epochs):
        print_r(0, f"Starting epoch {epoch + 1}/{args.num_train_epochs}")
        train_dataloader.sampler.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            if prof != None and global_step == args.profile_start:
                prof.start()
            if prof != None and global_step - args.profile_start == args.profile_steps:
                prof.stop()
                # print profile
                if dist.get_rank() == 0:
                    prof.export_chrome_trace("trace.json")
                    print(
                        prof.key_averages().table(
                            sort_by="self_cuda_time_total", row_limit=10
                        )
                    )
            step_start_time = time.time()
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            outputs = model_engine(**batch)
            loss = outputs.loss

            model_engine.backward(loss)
            model_engine.step()
            global_samples += model_engine.train_batch_size()

            step_time = time.time() - step_start_time
            if args.bench_start >= 0 and args.bench_steps > 0:
                if global_step >= args.bench_start:
                    total_time += step_time
                    total_count += 1
                if global_step >= args.bench_start + args.bench_steps - 1:
                    break

            if dist.get_rank() == 0 and args.wandb_name is not None:
                wandb.log({"global_samples": global_samples, "train-loss": loss})
            if global_step % 1 == 0:  # Print every step
                msg = f"Step {global_step}, Loss: {loss.item():.4f}, Time: {step_time * 1000:.0f}ms"
                print_r(0, msg)
                if dist.get_rank() == 0:
                    logging.info(msg)

            # Evaluation after every eval_steps
            if (
                args.eval_steps > 0
                and global_step % args.eval_steps == 0
                and save_checkpoint_p
            ):
                eval_loss = evaluate(model_engine, eval_dataloader)
                if dist.get_rank() == 0:
                    if eval_loss is not None:
                        eval_loss_val = float(eval_loss)
                        if args.wandb_name != None:
                            wandb.log(
                                {
                                    "global_samples": global_samples,
                                    "eval-loss": eval_loss_val,
                                }
                            )
                        eval_msg = f"[Eval @ step {global_step}] Eval Loss: {eval_loss_val:.4f}"
                        print(eval_msg, flush=True)
                        logging.info(eval_msg)
                    else:
                        eval_msg = f"[Eval @ step {global_step}] Eval Loss unavailable (no eval batches processed)"
                        print(eval_msg, flush=True)
                        logging.info(eval_msg)
            if (
                args.checkpoint_steps > 0
                and global_step > 0
                and global_step % args.checkpoint_steps == 0
                and save_checkpoint_p
            ):
                _save_weights(model_engine, tokenizer, args.output_dir, global_step)
            global_step += 1
            if prof != None:
                prof.step()
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        if args.bench_start >= 0 and args.bench_steps > 0:
            if global_step >= args.bench_start + args.bench_steps - 1:
                break
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    if args.bench_start >= 0 and args.bench_steps > 0:
        print_r(0, f"Average iteration time = {total_time / total_count}")

    if save_checkpoint_p:
        _save_weights(model_engine, tokenizer, args.output_dir, global_step)

    print_r(0, "Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )

    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--profile_start", type=int, default=-1)
    parser.add_argument("--profile_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bench_start", type=int, default=-1)
    parser.add_argument("--bench_steps", type=int, default=100)
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=0,
        help="Run evaluation every N steps (0 disables)",
    )
    parser.add_argument(
        "--max_length", type=int, default=2048, help="Max sequence length"
    )
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument(
        "--max_steps", type=int, default=-1, help="Stop after N steps (-1 = full epoch)"
    )
    parser.add_argument(
        "--checkpoint_steps", type=int, default=0,
        help="Save a checkpoint every N steps (0 disables); keeps last 2",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=4, help="Eval batch size per rank"
    )
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    main(args)
