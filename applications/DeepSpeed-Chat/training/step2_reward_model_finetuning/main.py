#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_critic_model
from utils.data.data_utils import create_prompt_dataset, DataCollatorReward
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument('--data_split',
                        type=str,
                        default='6,2,2',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `2,4,4`'
                        'will use 60% of data for phase 1, 20% for phase 2'
                        'and 20% for phase 3.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help='Where to store the data-related files such as shuffle index.')
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.1,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert (
            not args.only_optimize_lora
        ), "--gradient_checkpointing and --only_optimizer_lora cannot be enabled at the same time."

    return args


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    assert not args.offload, "zero-offload is not currently supported but coming soon!"

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token

    rm_model = create_critic_model(args.model_name_or_path, tokenizer,
                                   ds_config, args.num_padding_at_beginning)

    if args.lora_dim > 0:
        rm_model = convert_linear_layer_to_lora(rm_model,
                                                args.lora_module_name,
                                                args.lora_dim)
        if args.only_optimize_lora:
            rm_model = only_optimize_lora_parameters(rm_model)

    train_phase = 2
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_seq_len)

    # DataLoaders creation:
    data_collator = DataCollatorReward()
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    def evaluation_reward(model, eval_dataloader):
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        scores = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            chosen = outputs["chosen_mean_scores"]
            rejected = outputs["rejected_mean_scores"]
            correct_predictions += (chosen > rejected).sum()
            total_predictions += chosen.shape[0]
            scores += outputs["chosen_mean_scores"].mean().float()
            if step == 99:  # For faster evaluation and debugging
                break
            acc = correct_predictions / total_predictions
            scores = scores / (step + 1)
        try:
            acc = get_all_reduce_mean(acc).item()
            scores = get_all_reduce_mean(scores).item()
        except:
            pass
        return scores, acc

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        rm_model, args.weight_decay)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    rm_model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=rm_model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        rm_model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    print_rank_0(
        f"***** Evaluating reward, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    reward_score, acc = evaluation_reward(rm_model, eval_dataloader)
    print_rank_0(
        f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}",
        args.global_rank)

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        rm_model.train()
        mean_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = rm_model(**batch, use_cache=False)
            loss = outputs["loss"]
            rm_model.backward(loss)
            rm_model.step()
            mean_loss += loss.item()
        print_rank_0(
            f"Epoch {epoch+1}/{args.num_train_epochs} with loss {mean_loss/(step+1)}",
            args.global_rank)
        # Evaluate reward_loss on the validation set.
        print_rank_0(
            f"***** Evaluating reward, Epoch {epoch+1}/{args.num_train_epochs} *****",
            args.global_rank)
        reward_score, acc = evaluation_reward(rm_model, eval_dataloader)
        print_rank_0(
            f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}",
            args.global_rank)
        rm_model.tput_timer.update_epoch_count()

    if args.output_dir is not None:
        print_rank_0('saving model ...', args.global_rank)
        rm_model = convert_lora_to_linear_layer(rm_model)

        if args.global_rank == 0:
            save_hf_format(rm_model, tokenizer, args)
        if args.zero_stage == 3:
            # for zero stage 3, each gpu only has a part of the model, so we need to save the model on each gpu by using DS-Engine
            save_zero_three_model(rm_model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)


if __name__ == "__main__":
    main()
