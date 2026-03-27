import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, enable_full_determinism

from deepspeed.compile.passes.sp_compile import prepare_autosp_inputs

from distributed_attention import ulysses_attention_forward, set_padding_mask
# from ring_attention import ring_attention_forward
from sp_dp_registry import get_group, populate_registry, get_registry

torch.set_float32_matmul_precision("high")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = 12 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_args():
    parser = argparse.ArgumentParser(
        description="AutoSP benchmark script for distributed sequence parallel training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--seq_length", 
        type=int, 
        default=512,
        help="Sequence length for training"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=1,
        help="Total training steps"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=2e-5,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--activation_checkpointing", 
        action="store_true",
        help="Enable gradient checkpointing"
    )
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="timdettmers/openassistant-guanaco",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--num_layers", 
        type=int, 
        default=None,
        help="Number of transformer layers (None means use full model)"
    )
    
    # Compilation arguments
    parser.add_argument(
        "--compile", 
        type=str, 
        default="autosp",
        choices=["eager", "compile", "autosp", "ringattn"],
        help="Compilation mode: eager (no compilation), compile (torch.compile), autosp (AutoSP), ringattn (ring attention)"
    )
    parser.add_argument(
        "--backend", 
        type=str, 
        default="inductor",
        help="Backend compiler (e.g., inductor, cudagraph)"
    )
    
    parser.add_argument(
        "--deterministic", 
        action="store_true",
        help="Enable deterministic mode for reproducibility"
    )

    parser.add_argument(
        "--print_interval", 
        type=int, 
        default=1,
        help="Interval for printing metrics"
    )
    
    parser.add_argument(
        "--sp_size", 
        type=int, 
        default=2,
        help="Sequence parallel size"
    )
    parser.add_argument(
        "--dp_size", 
        type=int, 
        default=1,
        help="Data parallel size"
    )

    return parser.parse_args()

def validate_args(args):
    valid_compile_modes = ["eager", "compile", "autosp", "ringattn"]
    if args.compile not in valid_compile_modes:
        raise ValueError(
            f"Invalid compile mode: {args.compile}. "
            f"Must be one of {valid_compile_modes}"
        )
    
    if args.sp_size <= 0 or args.dp_size <= 0:
        raise ValueError("sp_size and dp_size must be positive integers")
    
    if args.seq_length <= 0:
        raise ValueError("seq_length must be positive")


def print_rank_0(accelerator, *args, **kwargs):
    """Print only on main process (rank 0)."""
    if accelerator.is_main_process:
        print(*args, **kwargs)


def main():
    args = get_args()
    validate_args(args)
    set_seed(12)

    if args.deterministic:
        enable_full_determinism(12)
        from torch._inductor import config
        config.fallback_random = True
        torch.use_deterministic_algorithms(True)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    device = accelerator.device
    assert accelerator.num_processes == args.sp_size * args.dp_size, 'Incorrect dp/sp sizing'

    print_rank_0(accelerator, "\n" + "="*60)
    print_rank_0(accelerator, "AutoSP Benchmark Configuration")
    print_rank_0(accelerator, "="*60)
    print_rank_0(accelerator, f"Model: {args.model_name}")
    print_rank_0(accelerator, f"Compile Mode: {args.compile}")
    print_rank_0(accelerator, f"Backend: {args.backend}")
    print_rank_0(accelerator, f"Sequence Parallel Size: {args.sp_size}")
    print_rank_0(accelerator, f"Data Parallel Size: {args.dp_size}")
    print_rank_0(accelerator, f"Total Processes: {accelerator.num_processes}")
    print_rank_0(accelerator, f"Batch Size: {args.batch_size}")
    print_rank_0(accelerator, f"Sequence Length: {args.seq_length}")
    print_rank_0(accelerator, f"Num Layers: {args.num_layers if args.num_layers else 'Full model'}")
    print_rank_0(accelerator, f"Deterministic: {args.deterministic}")
    print_rank_0(accelerator, f"Activation Checkpointing: {args.activation_checkpointing}")
    print_rank_0(accelerator, f"Learning Rate: {args.learning_rate}")
    print_rank_0(accelerator, f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
    print_rank_0(accelerator, "="*60 + "\n")

    ## Set sp/dp groups accordingly.
    if args.compile in ['compile', 'eager', 'ringattn']:
        populate_registry(args.sp_size, args.dp_size)

    print_rank_0(accelerator, "Loading model and tokenizer...")

    model_name = args.model_name
    if args.compile == "autosp":
        attention_backend = "sdpa"
    else:
        if args.compile == "eager" or args.compile == "compile":
            from transformers.models.llama import modeling_llama
            attention_backend = "ulyssess"
            modeling_llama.ALL_ATTENTION_FUNCTIONS["ulyssess"] = ulysses_attention_forward
        elif args.compile == "ringattn":
            from transformers.models.llama import modeling_llama
            attention_backend = "ringattn"
            modeling_llama.ALL_ATTENTION_FUNCTIONS["ringattn"] = ring_attention_forward 

    if args.num_layers is not None:
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        print_rank_0(accelerator, f"num_hidden_layers: {model_config.num_hidden_layers} -> {args.num_layers}")
        model_config.num_hidden_layers = args.num_layers
        model_config._attn_implementation = attention_backend
        model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
    else:
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_config._attn_implementation = attention_backend
        model = AutoModelForCausalLM.from_pretrained(model_name, config=model_config, trust_remote_code=True)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    print_rank_0(accelerator, "Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print_rank_0(accelerator, "Loading dataset...")

    g = torch.Generator()
    g.manual_seed(12)
    dataset = load_dataset('ag_news', split='train[:1%]')

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', max_length=args.seq_length, truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    num_replicas_ = args.dp_size
    rank_ = accelerator.process_index // args.sp_size

    sampler = DistributedSampler(tokenized_dataset, num_replicas=num_replicas_, rank=rank_, seed=12, shuffle=False)  
    data_loader = DataLoader(tokenized_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, worker_init_fn=seed_worker, generator=g)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
    print_rank_0(accelerator, f"Model prepared: {model.__class__}")

    if args.compile == "autosp":
        print_rank_0(accelerator, f"Running autosp with backend={args.backend}")
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.capture_scalar_outputs = True
        model.compile(backend=args.backend)
    elif args.compile in ["compile", "ringattn"]:
        print_rank_0(accelerator, f"Running torch.compile with backend={args.backend}")
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.capture_scalar_outputs = True
        model = torch.compile(model, backend=args.backend)
    else:
        print_rank_0(accelerator, f"Running eager")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = args.model_name.split("/")[-1]
    exp_name = f"{model_name}_np{accelerator.num_processes}_{args.compile}_" \
               f"B{args.backend}_" \
               f"L{0 if args.num_layers is None else args.num_layers}_" \
               f"bs{args.batch_size}_seq{args.seq_length}_" \
               f"T{timestamp}"
    
    model.train()
    global_step = 0
    print_rank_0(accelerator, f"Using global sequence length: {args.seq_length}")

    os.makedirs("logs", exist_ok=True)
    loss_log_file = open(f"logs/loss_{args.compile}_seq{args.seq_length}_rank{accelerator.process_index}.csv", "w")
    loss_log_file.write("step,loss\n")

    sp_rank = dist.get_rank() % args.sp_size
    for epoch in range(args.num_epochs):
        start_iter = time.time()

        for step, batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(device)
            B, S = input_ids.shape

            label_ids = input_ids.clone()
            position_ids = torch.arange(S, device=device).unsqueeze(0)
            attention_mask = batch['attention_mask'].to(device)

            if args.compile == 'autosp':
                # prepare inputs for autosp
                input_ids, label_ids, position_ids, attention_mask = prepare_autosp_inputs(
                    input_ids, label_ids, position_ids, attention_mask, seq_dim=1
                )
            else:
                chunk_size = S // args.sp_size
                start = sp_rank * chunk_size
                end = start + chunk_size
                input_ids = input_ids[:, start:end]
                label_ids = label_ids[:, start:end]
                position_ids = position_ids[:, start:end]

                # Store the padding mask to be accessed directly in local attention
                set_padding_mask(attention_mask)

            outputs = model(
                input_ids=input_ids,
                labels=label_ids,
                position_ids=position_ids,
                attention_mask=attention_mask
            )
            loss = outputs.loss

            elapsed_time = time.time() - start_iter
            alloc_mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

            if global_step % args.print_interval == 0:
                print(
                    f"[Rank {accelerator.process_index}] Epoch {epoch+1}, Step {global_step}, Loss: {loss.item():.4f}, "
                    f"Time: {elapsed_time:.2f}s, "
                    f"Alloc Mem: {alloc_mem_gb:.2f} GB, "
                    f"Peak Mem: {peak_mem_gb:.2f} GB"
                )

            accelerator.backward(loss)

            loss_log_file.write(f"{global_step},{loss.item()}\n")
            loss_log_file.flush()

            global_step += 1
            if global_step > args.steps:
                break

if __name__ == "__main__":
    torch._dynamo.config.accumulated_cache_size_limit = 256
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False
    
    main()

