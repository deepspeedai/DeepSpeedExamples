import torch
import time
import deepspeed
import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator
)
import random
import numpy as np
from deepspeed import comm as dist

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preprocess_alpaca(example, tokenizer, max_length=512):
    prompt = f"### Instruction:\n{example['instruction']}\n\n"
    if example.get("input", ""):
        prompt += f"### Input:\n{example['input']}\n\n"
    prompt += f"### Response:\n{example['output']}"
    tokenized = tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def main(args):
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)

    # Load Alpaca 52K dataset
    dataset = load_dataset("tatsu-lab/alpaca")

    tokenized_dataset = dataset["train"].map(lambda x: preprocess_alpaca(x, tokenizer), batched=False)
    
    # Create DataLoader - let DeepSpeed handle the actual batching
    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=1,  # This will be overridden by DeepSpeed config
        collate_fn=default_data_collator,
        shuffle=True
    )

    # DeepSpeed will automatically parse the config file passed via --deepspeed argument
    model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=tokenized_dataset,
        collate_fn=default_data_collator
    )

    model_engine.train()
    global_step = 0

    for epoch in range(args.num_train_epochs):
        if dist.get_rank() == 0:
            print(f"Starting epoch {epoch + 1}/{args.num_train_epochs}")
        
        for step, batch in enumerate(train_dataloader):
            step_start_time = time.time()
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            outputs = model_engine(**batch)
            loss = outputs.loss

            model_engine.backward(loss)
            model_engine.step()

            step_time = time.time() - step_start_time
            global_step += 1
            
            if dist.get_rank() == 0:  # Print every 10 steps
                print(f"Step {global_step}, Loss: {loss.item():.4f}, Time: {step_time*1000:.0f}ms")

    # Save model using DeepSpeed's save_checkpoint method
    if dist.get_rank() == 0:
        model_engine.save_checkpoint(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument('--local_rank',
                    type=int,
                    default=-1,
                    help='local rank passed from distributed launcher')
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    main(args)