#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import random
import time
from pathlib import Path
from re import L

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
# from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

import deepspeed

from deepspeed.runtime.data_pipeline.data_routing.helper import convert_to_random_ltd, save_without_random_ltd

import numpy as np
from learning_rates import AnnealingLR


logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="constant_with_warmup",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=None, help="lr schedule warmup ratio."
    )
    parser.add_argument(
        "--decay_style", type=str, default=None, help="lr schedule decay style."
    )
    parser.add_argument(
        "--token_based_lr_decay", action="store_true", help="Use token-based LR decay"
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--not_tie_wre", action="store_true", help="tie the last layer and embedding or not."
    )
    parser.add_argument("--random_ltd", action="store_true", help="enable random-ltd or not."
    )
    parser.add_argument("--curriculum_learning", action="store_true", help="enable curriculum learning or not."
    )
    parser.add_argument("--eval_step", type=int, default=10, help="eval step."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--data_folder", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()
    def print_rank_0(msg):
        if args.local_rank <= 0:
            print(msg)
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.distributed.barrier()

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )

    
    if args.model_name_or_path is not None:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    if args.not_tie_wre:
        config.tie_word_embeddings=False    


    if args.model_name_or_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        print_rank_0("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))


    model.to(device)
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)
    args.block_size = block_size

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, collate_fn=default_data_collator, sampler=train_sampler, batch_size=args.per_device_train_batch_size
    )
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, sampler=eval_sampler, batch_size=args.per_device_eval_batch_size
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        
    # Train!
    print_rank_0("***** Running training *****")
    print_rank_0(f"  Num examples = {len(train_dataset)}")
    print_rank_0(f"  Num Epochs = {args.num_train_epochs}")
    print_rank_0(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print_rank_0(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print_rank_0(f"  Total optimization steps = {args.max_train_steps}")
    print_rank_0(f"  Block size (seqlen) = {args.block_size}")


    num_p = sum([p.numel() for p in model.parameters()])
    print_rank_0('Number of parameters: {}'.format(num_p))

    def to_device(batch):
        output = {}
        for k, v in batch.items():
            try:
                output[k] = v.to(device)
            except:
                output[k] = v
        return output

    def evaluation(model, eval_dataloader):
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            # batch = tuple(t.to(device) for t in batch)
            batch = to_device(batch)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(loss.cpu().item())
        losses = losses[: len(eval_dataset)]
        try:
            perplexity = math.exp(np.mean(losses))
        except OverflowError:
            perplexity = float("inf")
        return perplexity

    def data_post_process(data, data_sampler_state_dict):
        print(data)
        if 'seqlen_truncate' in data_sampler_state_dict['current_difficulties']:
            args.data_efficiency_curriculum_learning_seqlen_type = 'seqlen_truncate'
            current_seqlen = data_sampler_state_dict['current_difficulties']['seqlen_truncate']
            if current_seqlen < args.block_size:
                data['attention_mask'] = data['attention_mask'][:, :current_seqlen].contiguous()
                data['input_ids'] = data['input_ids'][:, :current_seqlen].contiguous()
                data['labels'] = data['labels'][:, :current_seqlen].contiguous()
        elif 'seqlen_reshape' in data_sampler_state_dict['current_difficulties']:
            args.data_efficiency_curriculum_learning_seqlen_type = 'seqlen_reshape'
            current_seqlen = data_sampler_state_dict['current_difficulties']['seqlen_reshape']
            if current_seqlen < args.block_size:
                orig_num_token = torch.numel(data['input_ids'])
                reshape_len = (data['input_ids'].size()[1] // current_seqlen) * current_seqlen
                data['input_ids'] = torch.cat((data['input_ids'][:, :reshape_len].contiguous().view(-1, current_seqlen),
                    data['input_ids'][:, -(current_seqlen):]), 0).contiguous()
                data['attention_mask'] = torch.cat((data['attention_mask'][:, :reshape_len].contiguous().view(-1, current_seqlen),
                    data['attention_mask'][:, -(current_seqlen):]), 0).contiguous()
                data['labels'] = torch.cat((data['labels'][:, :reshape_len].contiguous().view(-1, current_seqlen),
                    data['labels'][:, -(current_seqlen):]), 0).contiguous()
                num_row = math.ceil(orig_num_token / current_seqlen)
                num_row = min(num_row, data['input_ids'].size()[0])
                data['input_ids'] = data['input_ids'][:num_row, :].contiguous()
                data['attention_mask'] = data['attention_mask'][:num_row, :].contiguous()
                data['labels'] = data['labels'][:num_row, :].contiguous()
        else:
            args.data_efficiency_curriculum_learning_seqlen_type = None
        return data

    def training(model, train_dataloader, train_dataset, eval_dataloader, num_train_epochs, args):
        start = time.time()
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        world_size = torch.distributed.get_world_size()
        if args.warmup_ratio is not None:
            args.num_warmup_steps = int(args.max_train_steps*args.warmup_ratio)
        print_rank_0 (f"world_size {world_size} num_warmup_steps {args.num_warmup_steps}")
        total_tokens = args.max_train_steps*args.per_device_train_batch_size*args.gradient_accumulation_steps*args.block_size*world_size
        
        if args.token_based_lr_decay:
            lr_scheduler = AnnealingLR(
                optimizer,
                max_lr=args.learning_rate,
                min_lr=0,
                warmup_steps=args.num_warmup_steps,
                decay_tokens=total_tokens,
                decay_style=args.decay_style)
        else:
            lr_scheduler = get_scheduler(
                name=args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=args.num_warmup_steps,
                num_training_steps=args.max_train_steps,
            )
        
        if args.curriculum_learning:
            model, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                args=args,
                lr_scheduler=lr_scheduler,
                training_data=train_dataset,
                collate_fn=default_data_collator,
                dist_init_required=True)
            model.set_data_post_process_func(data_post_process)
        else:
            model, optimizer, _, lr_scheduler = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                args=args,
                lr_scheduler=lr_scheduler,
                dist_init_required=True)
        if args.random_ltd:
            model = convert_to_random_ltd(model, GPT2Block)
            random_ltd_layer_num = model.random_ltd_scheduler.get_random_ltd_layer_num()
            total_layer_num = model.random_ltd_scheduler.model_layer_num
           
        epoch = 0
        global_step = 0
        micro_step = 0
        current_best = float("inf")
        consumed_token = 0
        args.eval_step = max(1, args.max_train_steps // 100)
        while consumed_token < total_tokens:
        # for epoch in range(num_train_epochs):
            if epoch == 0:
                perplexity = evaluation(model, eval_dataloader)
                current_best = min(current_best, perplexity)
                print_rank_0 (f"*************************initialization with perplexity {perplexity}***********************************")            
            model.train()
            for step, batch in enumerate(train_dataloader):
                model.train()
                if args.curriculum_learning:
                    curriculum_seqlen = batch['input_ids'].size()[1]
                    if hasattr(args, 'data_efficiency_curriculum_learning_seqlen_type') and \
                        args.data_efficiency_curriculum_learning_seqlen_type == 'seqlen_reshape':
                        args.data_efficiency_curriculum_learning_numel = torch.numel(batch['input_ids'])
                batch = to_device(batch)                
                outputs = model(**batch)
                loss = outputs.loss
                # loss = loss / args.gradient_accumulation_steps # DeepSpeed engine will handle this loss scaling (_scale_loss_by_gas), thus no need to do so on user side
                model.backward(loss)

                actual_seq_length = args.block_size
                if args.curriculum_learning:
                    actual_seq_length = curriculum_seqlen
                if args.random_ltd:
                    reserved_length = model.random_ltd_scheduler.get_current_seq()
                    if reserved_length < actual_seq_length:
                        actual_seq_length = (actual_seq_length * (total_layer_num - random_ltd_layer_num) + reserved_length * random_ltd_layer_num) // total_layer_num
                if args.curriculum_learning:
                    if hasattr(args, 'data_efficiency_curriculum_learning_numel'):
                        act_mbsz = args.data_efficiency_curriculum_learning_numel / curriculum_seqlen
                        act_token = act_mbsz * actual_seq_length
                        consumed_token += act_token * world_size
                    else:
                        consumed_token += actual_seq_length * args.per_device_train_batch_size * world_size
                else:
                    consumed_token += actual_seq_length * args.per_device_train_batch_size * world_size

                if args.token_based_lr_decay:
                    model.step(lr_kwargs={'increment': 1, 'consumed_tokens': consumed_token})
                else:
                    model.step()
                micro_step += 1
                if micro_step % args.gradient_accumulation_steps == 0:
                    global_step += 1
                    # Evaluate perplexity on the validation set.
                    if global_step%args.eval_step==0 or step == len(train_dataloader)-1:
                        perplexity = evaluation(model, eval_dataloader)
                        current_best = min(current_best, perplexity)
                        log_text = f"At epoch {epoch+1} step {global_step} consumed_token {consumed_token} perplexity {perplexity} current best {current_best}"
                        if args.random_ltd:
                            log_text = f"{log_text} random-ltd reserved_length {reserved_length}"
                        print_rank_0(log_text)
                if consumed_token >= total_tokens:
                    break
            perplexity = evaluation(model, eval_dataloader)
            current_best = min(current_best, perplexity)
            print_rank_0(f"End of epoch {epoch+1} step {global_step} consumed_token {consumed_token} perplexity {perplexity} current best {current_best}")
            if consumed_token >= total_tokens:
                break
            epoch += 1
        duration = (time.time() - start) / 3600.0
        print_rank_0(f"End of training epoch {epoch+1} step {global_step} consumed_token {consumed_token} best perplexity {current_best} time {duration} hr")
        if args.output_dir is not None:
            print_rank_0('saving model ...')
            if not os.path.isdir(args.output_dir):
                os.makedirs(args.output_dir)

            if torch.distributed.get_rank() == 0:
                model_to_save = model.module if hasattr(model, 'module') else model
                CONFIG_NAME = "config.json"
                WEIGHTS_NAME = "pytorch_model.bin"
                output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                torch.save(save_without_random_ltd(model_to_save), output_model_file)
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(args.output_dir)


    training(model, train_dataloader, train_dataset, eval_dataloader, args.num_train_epochs, args)
    
if __name__ == "__main__":
    main()
