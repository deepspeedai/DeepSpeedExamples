#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Fine-tuning script for language models using DeepSpeed ZeRO Stage 3.
"""

import argparse
import os
import time
import logging
from datetime import datetime
from typing import Dict, Any

import torch
import deepspeed
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (AutoModelForCausalLM, AutoTokenizer, default_data_collator, set_seed,
                          enable_full_determinism)
from deepspeed import comm as dist

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_logger(rank: int = 0, log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("finetune_zero3")
    logger.handlers.clear()
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    if rank == 0:
        handler = logging.StreamHandler()
        handler.setLevel(numeric_level)
        formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Constants
DEFAULT_OPTIMIZER_BETAS = (0.9, 0.999)
MS_PER_SECOND = 1000
TFLOPS_DENOMINATOR = 1e12

# Alpaca dataset formatting
ALPACA_INSTRUCTION_TEMPLATE = "### Instruction:\n{instruction}\n\n"
ALPACA_INPUT_TEMPLATE = "### Input:\n{input}\n\n"
ALPACA_RESPONSE_TEMPLATE = "### Response:\n{output}"


def get_parameter_count(parameter: torch.nn.Parameter) -> int:
    return parameter.ds_numel if hasattr(parameter, "ds_tensor") else parameter.numel()


def estimate_transformer_tflops(seq_len: int,
                                model_size: int,
                                num_layers: int,
                                hidden_size: int,
                                use_activation_checkpointing: bool = False) -> float:
    """
    Estimate TFLOPS for decoder-only dense models.
    """
    coefficient = 4 if use_activation_checkpointing else 3
    tflops = (2 * coefficient * model_size * seq_len +
              2 * 2 * coefficient * num_layers * hidden_size * seq_len**2) / TFLOPS_DENOMINATOR
    return tflops


def preprocess_alpaca_example(example: Dict[str, str],
                              tokenizer: AutoTokenizer,
                              max_length: int = 2048) -> Dict[str, Any]:
    prompt = ALPACA_INSTRUCTION_TEMPLATE.format(instruction=example['instruction'])

    if example.get("input", "").strip():
        prompt += ALPACA_INPUT_TEMPLATE.format(input=example['input'])

    prompt += ALPACA_RESPONSE_TEMPLATE.format(output=example['output'])

    tokenized = tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length", return_tensors=None)

    # Mask padding out of the loss (HF CrossEntropy ignore_index = -100); otherwise the ~max_length
    # padding dominates the loss. attention_mask is 0 exactly on padding positions.
    tokenized["labels"] = [
        token if mask == 1 else -100 for token, mask in zip(tokenized["input_ids"], tokenized["attention_mask"])
    ]

    return tokenized


def detect_moe_model(model: AutoModelForCausalLM) -> bool:
    moe_config_attrs = ['num_local_experts', 'moe_layers', 'num_experts', 'expert_capacity', 'router_aux_loss_coef']

    for attr in moe_config_attrs:
        if hasattr(model.config, attr):
            return True
    return False


def create_experiment_name(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_short = args.model_name.split("/")[-1]
    activation_checkpointing = 1 if args.activation_checkpointing else 0

    exp_name = (f"{model_name_short}_bs{args.batch_size}_seq{args.max_length}"
                f"_ac{activation_checkpointing}_T{timestamp}")
    return exp_name


def load_tokenizer(model_name: str, logger: logging.Logger) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    return tokenizer


def load_model(model_name: str, attn_implementation: str, logger: logging.Logger) -> AutoModelForCausalLM:
    logger.debug(f"Loading model: {model_name}")
    logger.debug(f"Attention implementation: {attn_implementation}")

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.bfloat16,
                                                 attn_implementation=attn_implementation)

    return model


def setup_model_training(model: torch.nn.Module,
                         use_activation_checkpointing: bool = True,
                         logger: logging.Logger = None) -> None:
    if use_activation_checkpointing:
        if logger:
            logger.debug("Enabling gradient checkpointing...")
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


def create_optimizer(model: AutoModelForCausalLM, args: argparse.Namespace) -> Any:
    # With Reflow enabled (a "reflow" block), DeepSpeed transparently remaps the CPU optimizer below to
    # its ReflowCPUAdam / ReflowCPULion subclass, so the same client optimizer drives both the
    # ZeRO-Offload baseline and Reflow.
    if args.optimizer == "lion":
        from deepspeed.ops.lion import DeepSpeedCPULion
        optimizer = DeepSpeedCPULion(
            model.parameters(),
            lr=args.lr,
            betas=DEFAULT_OPTIMIZER_BETAS,
            weight_decay=args.weight_decay,
        )
    else:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        optimizer = DeepSpeedCPUAdam(
            model.parameters(),
            lr=args.lr,
            betas=DEFAULT_OPTIMIZER_BETAS,
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
    return optimizer


def load_and_preprocess_dataset(dataset_name: str, dataset_percentage: float, tokenizer: AutoTokenizer,
                                max_length: int, logger: logging.Logger) -> Any:
    logger.debug(f"Loading dataset: {dataset_name}")

    dataset = load_dataset(dataset_name)
    original_size = len(dataset["train"])

    if dataset_percentage < 100.0:
        subset_size = int(original_size * dataset_percentage / 100.0)
        if subset_size == 0:
            raise ValueError("dataset_percentage selects no training examples")
        dataset["train"] = dataset["train"].select(range(subset_size))
        logger.debug(f"Using {dataset_percentage}% of dataset: {subset_size}/{original_size} examples")
    else:
        logger.debug(f"Using full dataset: {original_size} examples")

    logger.debug("Tokenizing dataset...")

    tokenized_dataset = dataset["train"].map(lambda x: preprocess_alpaca_example(x, tokenizer, max_length),
                                             batched=False,
                                             remove_columns=dataset["train"].column_names,
                                             desc="Tokenizing")

    return tokenized_dataset


def initialize_wandb(args: argparse.Namespace, exp_name: str, logger: logging.Logger) -> None:
    if args.use_wandb and dist.get_rank() == 0:
        try:
            wandb_run_name = args.wandb_run_name if args.wandb_run_name else exp_name
            logger.debug(f"Initializing WandB run: {wandb_run_name}")
            wandb.init(project=args.wandb_project, name=wandb_run_name, tags=args.wandb_tags, config=vars(args))
            logger.debug("WandB initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
            args.use_wandb = False


def report_cpu_simd(logger: logging.Logger) -> None:
    """Log the CPU SIMD level used by the Reflow CPU-Adam kernel."""
    flags = set()
    try:
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("flags") or line.startswith("Features"):
                    flags = set(line.split(":", 1)[1].split())
                    break
    except OSError:
        logger.debug("Could not read /proc/cpuinfo; skipping CPU SIMD report")
        return

    has_avx512 = "avx512f" in flags
    has_avx2 = "avx2" in flags
    level = "AVX-512" if has_avx512 else ("AVX-256" if has_avx2 else "scalar (no AVX)")
    logger.info(f"Reflow CPU-Adam SIMD: {level}")
    if not has_avx512 and not has_avx2:
        logger.warning("No AVX: BF16 CPU-Adam uses the scalar fallback (much slower).")


def main(args: argparse.Namespace) -> None:
    logger = setup_logger(rank=0, log_level=args.log_level)

    exp_name = create_experiment_name(args)

    logger.debug(f"Starting experiment: {exp_name}")
    logger.debug("Training configuration:")
    logger.debug(f"  Model: {args.model_name}")
    logger.debug(f"  Batch size: {args.batch_size}")
    logger.debug(f"  Max length: {args.max_length}")
    logger.debug(f"  Learning rate: {args.lr}")
    logger.debug(f"  Epochs: {args.num_train_epochs}")
    logger.debug(f"  Activation checkpointing: {args.activation_checkpointing}")

    tokenizer = load_tokenizer(args.model_name, logger)
    model = load_model(args.model_name, args.attn_implementation, logger)
    if args.leaf_module:
        from deepspeed.utils import set_z3_leaf_modules
        logger.debug(f"Setting leaf_module to: {args.leaf_module}")
        set_z3_leaf_modules(model, [args.leaf_module])
    setup_model_training(model, args.activation_checkpointing, logger)
    optimizer = create_optimizer(model, args)

    tokenized_dataset = load_and_preprocess_dataset(args.dataset_name, args.dataset_percentage, tokenizer,
                                                    args.max_length, logger)

    # Initialize DeepSpeed. We build the data loader ourselves (below) instead of passing
    # training_data, so the shuffle order is reproducible across runs.
    model_engine, optimizer, _, _ = deepspeed.initialize(args=args, model=model, optimizer=optimizer)

    # Deterministic data order: a DistributedSampler with a fixed seed shards the dataset across
    # ranks and shuffles reproducibly, so two runs (e.g. reflow vs zerooffload) see the identical
    # batch at every step. DeepSpeed's built-in training_data loader does not pin the shuffle seed.
    train_sampler = DistributedSampler(tokenized_dataset,
                                       num_replicas=dist.get_world_size(),
                                       rank=dist.get_rank(),
                                       shuffle=True,
                                       seed=args.seed)
    train_dataloader = DataLoader(tokenized_dataset,
                                  batch_size=model_engine.train_micro_batch_size_per_gpu(),
                                  sampler=train_sampler,
                                  collate_fn=default_data_collator)

    logger = setup_logger(rank=dist.get_rank(), log_level=args.log_level)
    report_cpu_simd(logger)

    initialize_wandb(args, exp_name, logger)

    model_engine.train()

    sequence_length = args.max_length
    model_size = sum(get_parameter_count(p) for p in model.parameters())
    is_moe_model = detect_moe_model(model)

    logger.debug(f"Model type: {'MoE' if is_moe_model else 'Dense'}")
    logger.debug(f"Model size: {model_size:,} parameters")

    # Calculate TFLOPS only for non-MoE models. Two figures are tracked:
    #   tflops            -> counts the activation-recompute forward (hardware FLOPs actually executed)
    #   effective tflops  -> excludes recompute (useful model FLOPs only: 1 forward + 2 backward)
    total_tflops = None
    total_tflops_effective = None
    if not is_moe_model:
        total_tflops = estimate_transformer_tflops(sequence_length, model_size, model.config.num_hidden_layers,
                                                   model.config.hidden_size, args.activation_checkpointing)
        total_tflops_effective = estimate_transformer_tflops(sequence_length,
                                                             model_size,
                                                             model.config.num_hidden_layers,
                                                             model.config.hidden_size,
                                                             use_activation_checkpointing=False)

    global_step = 0
    iter_times = []
    losses = []

    stop = False
    for epoch in range(args.num_train_epochs):
        train_sampler.set_epoch(epoch)
        logger.debug(f"Starting epoch {epoch + 1}/{args.num_train_epochs}")

        for batch in train_dataloader:
            step_start_time = time.perf_counter()
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}

            actual_batch_size = batch['input_ids'].shape[0]
            global_batch_size = actual_batch_size * dist.get_world_size()
            tokens_in_batch = global_batch_size * sequence_length

            outputs = model_engine(**batch)
            loss = outputs.loss

            model_engine.backward(loss)

            model_engine.step()

            # Include GPU completion rather than timing only asynchronous launches.
            _raw_loss = loss.item()
            step_time = time.perf_counter() - step_start_time
            global_step += 1

            if global_step > args.warmup_steps:
                iter_times.append(step_time)

            losses.append(_raw_loss)
            if args.loss_check:
                # Full-precision (bit-exact hex) per-step loss so a reflow run and a zerooffload run
                # under loss-check mode can be diffed bit-for-bit.
                logger.info(f"BITLOSS step {global_step} hex={float(_raw_loss).hex()} dec={_raw_loss:.12e}")

            tokens_per_second = tokens_in_batch / step_time
            step_tflops = None
            effective_tflops = None

            if not is_moe_model and total_tflops is not None:
                step_tflops = global_batch_size * total_tflops / step_time
                effective_tflops = global_batch_size * total_tflops_effective / step_time

            if global_step % args.log_interval == 0:
                avg_loss = sum(losses[-args.log_interval:]) / len(losses[-args.log_interval:])

                if is_moe_model:
                    # Skip throughput metrics for MoE models
                    log_msg = (f"Step {global_step:4d} | "
                               f"Loss: {avg_loss:.4f} | "
                               f"Time: {step_time * MS_PER_SECOND:5.0f}ms")
                else:
                    log_msg = (f"Step {global_step:4d} | "
                               f"Loss: {avg_loss:.4f} | "
                               f"Time: {step_time * MS_PER_SECOND:5.0f}ms | "
                               f"TFLOPS(w/ recompute): {step_tflops:8.2f} | "
                               f"effective TFLOPS(w/o recompute): {effective_tflops:8.2f} | "
                               f"Tokens/s: {tokens_per_second:6.0f}")

                logger.info(log_msg)

                if args.use_wandb and dist.get_rank() == 0:
                    log_dict = {
                        "train/loss": avg_loss,
                        "train/epoch": epoch + 1,
                        "train/global_step": global_step,
                        "train/learning_rate": args.lr,
                        "perf/step_time_ms": step_time * MS_PER_SECOND,
                        "perf/tokens_per_second": tokens_per_second,
                    }

                    if not is_moe_model and step_tflops is not None:
                        log_dict["perf/tflops"] = step_tflops  # with recompute (hardware)
                        log_dict["perf/effective_tflops"] = effective_tflops  # without recompute (useful)

                    wandb.log(log_dict, step=global_step)

            stop = global_step >= args.bench_steps
            if stop:
                break

        if stop:
            break

    if iter_times:
        mean_step_ms = sum(iter_times) / len(iter_times) * MS_PER_SECOND
        logger.info(f"Mean step time after {args.warmup_steps} warmup steps: {mean_step_ms:.2f}ms")

    if args.save_checkpoint:
        # ZeRO checkpoint saving includes collectives, so every rank must participate.
        model_engine.save_checkpoint(args.output_dir)
        if dist.get_rank() == 0:
            tokenizer.save_pretrained(args.output_dir)

    if args.use_wandb and dist.get_rank() == 0:
        try:
            wandb.finish()
            logger.debug("WandB run finished successfully")
        except Exception as e:
            logger.error(f"Error finishing WandB run: {e}")

    logger.debug("Training completed successfully!")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="Fine-tune language models with DeepSpeed ZeRO Stage 3",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, required=True, help="Training batch size per device")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")

    parser.add_argument("--attn_implementation",
                        type=str,
                        default="flash_attention_2",
                        choices=["eager", "sdpa", "flash_attention_2"],
                        help="Attention implementation to use")
    parser.add_argument("--leaf_module",
                        type=str,
                        default=None,
                        help="Set leaf_module to enable fine-tuning MoE models")
    parser.add_argument("--activation_checkpointing",
                        action="store_true",
                        help="Enable activation checkpointing to save memory")
    parser.add_argument("--optimizer",
                        type=str,
                        default="adam",
                        choices=["adam", "lion"],
                        help="CPU optimizer to offload (Reflow remaps it to its async subclass)")

    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length for tokenization")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimization")

    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed from distributed launcher")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--deterministic",
                        action="store_true",
                        help="Enable deterministic training for full reproducibility")
    parser.add_argument("--loss_check",
                        action="store_true",
                        help="Bit-level loss-check mode (verification only): full determinism including "
                        "the NCCL collectives (fixed algorithm/protocol) and cuBLAS, plus per-step "
                        "bit-exact loss logging, so two configs (e.g. reflow vs zerooffload) can be "
                        "diffed bit-for-bit. Not for throughput runs.")

    parser.add_argument("--log_interval", type=int, default=1, help="Log performance metrics every N steps")
    parser.add_argument("--log_level",
                        type=str,
                        default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level for controlling output verbosity")
    parser.add_argument("--warmup_steps",
                        type=int,
                        default=15,
                        help="Number of warmup steps for performance measurements")
    parser.add_argument("--bench_steps", type=int, default=100, help="Number of benchmark steps to run")

    parser.add_argument("--save_checkpoint", action="store_true", help="Save model checkpoint after training")

    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="reflow", help="WandB project name")
    parser.add_argument("--wandb_run_name",
                        type=str,
                        default=None,
                        help="WandB run name (auto-generated if not provided)")
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=[], help="WandB tags for the run")

    parser.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca", help="HuggingFace dataset name")
    parser.add_argument("--dataset_percentage",
                        type=float,
                        default=100.0,
                        help="Percentage of dataset to use (1.0-100.0)")

    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.dataset_percentage <= 0 or args.dataset_percentage > 100:
        raise ValueError("dataset_percentage must be between 0 and 100")

    if args.max_length <= 0:
        raise ValueError("max_length must be positive")

    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if args.lr <= 0:
        raise ValueError("learning rate must be positive")

    if args.bench_steps <= 0 or not 0 <= args.warmup_steps < args.bench_steps:
        raise ValueError("bench_steps must be positive and warmup_steps must be below bench_steps")

    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive")

    if args.num_train_epochs <= 0:
        raise ValueError("num_train_epochs must be positive")


if __name__ == "__main__":
    parser = create_argument_parser()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    validate_arguments(args)

    if args.loss_check:
        # Bit-level loss-check mode: make the ENTIRE pipeline deterministic -- including the NCCL
        # collectives (fixed algorithm/protocol) and cuBLAS -- so two configs (e.g. reflow vs
        # zerooffload) can be diffed bit-for-bit. Throughput is irrelevant here. These env vars must
        # be set before any CUDA/NCCL init (i.e. before deepspeed.initialize), which is why this runs
        # in __main__ ahead of main().
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ.setdefault("NCCL_ALGO", "Ring")
        os.environ.setdefault("NCCL_PROTO", "Simple")
        enable_full_determinism(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        logging.basicConfig(level=getattr(logging, args.log_level.upper()))
        logging.info("Loss-check mode: full determinism (NCCL Ring/Simple + cuBLAS + deterministic "
                     "algorithms) and bit-exact loss logging enabled")
    elif args.deterministic:
        enable_full_determinism(args.seed)
        torch.backends.cudnn.benchmark = False
        logging.basicConfig(level=getattr(logging, args.log_level.upper()))
        logging.info("Enabled deterministic mode for full reproducibility")
    else:
        set_seed(args.seed)
        logging.basicConfig(level=getattr(logging, args.log_level.upper()))
        logging.info(f"Set random seed to {args.seed}")

    main(args)
