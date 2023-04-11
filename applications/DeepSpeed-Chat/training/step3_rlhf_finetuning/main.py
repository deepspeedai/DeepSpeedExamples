#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""

engine = DeepSpeedRLHFEngine(actor_model_name_or_path=actor_model_name_or_path,
                             critic_model_name_or_path=critic_model_name_or_path,
                             tokenizer=tokenizer,
                             args=args)
trainer = DeepSpeedPPOTrainer(engine=engine, args=args)

for prompt_batch in prompt_train_dataloader:
    out = trainer.generate_experience(prompt_batch)
    actor_loss, critic_loss = trainer.train_rlhf(out)

"""
import argparse
import os
import random
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
)

import deepspeed

from ppo_trainer import DeepSpeedPPOTrainer, DeepSpeedPPOTrainerUnsupervised
from rlhf_engine import DeepSpeedRLHFEngine

import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset, MiniDataset, DataCollatorRLHF, get_unsupervised_data
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, moving_average, save_zero_three_model
from utils.module.lora import convert_lora_to_linear_layer


def parse_args():
    parser = argparse.ArgumentParser(
        description="(Step 3) RLHF training arguments")

    parser.add_argument(
        '--data_path',
        nargs='*',
        default=['Dahoas/rm-static'],
        help=
        'Path to the training dataset. Accepted format: 1) a single data path, 2) multiple datasets in the form: dataset1-path dataset2-path ...'
    )
    parser.add_argument(
        '--data_split',
        type=str,
        default='6,2,2',
        help=
        'Comma-separated list of proportions for training phase 1, 2, and 3 data. For example the split `2,4,4` '
        'will use 60% of data for phase 1, 20% for phase 2 and 20% for phase 3.'
    )
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--unsupervised_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).")
    parser.add_argument(
        "--unsupervised_dataset_config_name",
        type=str,
        default=None,
        help=
        "The configuration name of the dataset to use (via the datasets library)."
    )
    parser.add_argument("--unsup_coef",
                        type=float,
                        default=27.8,
                        help='''gamma in Equation 2 from InstructGPT paper''')
    parser.add_argument(
        "--actor_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True)
    parser.add_argument(
        "--critic_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True)
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. We did not see this in other models but keep it as an option for now."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help=
        "Batch size (per device) for the training dataloader and generation purpose."
    )
    parser.add_argument(
        "--per_device_mini_train_batch_size",
        type=int,
        default=16,
        help=
        "Mini Batch size (per device) for the training dataloader and training purpose."
    )
    parser.add_argument("--generation_batch_numbers",
                        type=int,
                        default=1,
                        help="Generate x batches to go to training mode.")
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=1,
        help="For generated data, how many ppo training epochs to run.")
    parser.add_argument("--max_prompt_seq_len",
                        type=int,
                        default=256,
                        help="The maximum sequence length.")
    parser.add_argument("--max_answer_seq_len",
                        type=int,
                        default=256,
                        help="The maximum sequence length.")
    parser.add_argument(
        "--actor_learning_rate",
        type=float,
        default=9.65e-6,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument(
        "--critic_learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument("--actor_weight_decay",
                        type=float,
                        default=0.1,
                        help="Weight decay to use.")
    parser.add_argument("--critic_weight_decay",
                        type=float,
                        default=0.1,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
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
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    # DeepSpeed
    parser.add_argument(
        "--enable_hybrid_engine",
        action='store_true',
        help=
        "Enable hybrid engine for actor model to optimize both inference and training through DeepSpeed."
    )
    parser.add_argument(
        "--unpin_actor_parameters",
        action='store_true',
        help=
        "Unpin actor's parameters during generation. This makes generation slower but requires less memory."
    )
    parser.add_argument(
        "--release_inference_cache",
        action='store_true',
        help=
        "Release the memory cache used for inference. This makes generation preparation slower but might increase e2e throughput by using larger batch size."
    )
    parser.add_argument(
        "--inference_tp_size",
        type=int,
        default=1,
        help=
        "Tensor-parallelism degree used for the inference-optimization. Please note hybrid-engine need to be enabled when using this feature."
    )
    parser.add_argument(
        "--tp_gather_partition_size",
        type=int,
        default=8,
        help=
        "Granularity to bring in layers for TP sharding inside the hybrid engine. Please note hybrid-engine and tp_inference_size > 1 need to be true when using this feature."
    )
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--offload_reference_model',
        action='store_true',
        help='Enable ZeRO Offload techniques for reference model')
    parser.add_argument(
        '--actor_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    parser.add_argument(
        '--critic_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Critic model (and reward).')
    parser.add_argument(
        '--actor_gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')
    parser.add_argument(
        '--critic_gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Critic model.')
    ## LoRA for efficient training setting
    parser.add_argument("--actor_lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--actor_lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument("--critic_lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--critic_lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    ## Make EMA as an optional feature
    parser.add_argument('--enable_ema',
                        action='store_true',
                        help='Enable EMA checkpoint for the model.')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if (args.actor_gradient_checkpointing
            and args.actor_lora_dim > 0) or (args.critic_gradient_checkpointing
                                             and args.critic_lora_dim > 0):
        assert (
            not args.only_optimize_lora
        ), "--{actor,critic}_gradient_checkpointing and --only_optimizer_lora cannot be enabled at the same time."

    if args.inference_tp_size > 1:
        assert (
            args.actor_zero_stage == 3
        ), "Zero stage 3 must be used to do Tensor sharding in the hybrid engine"

    return args


def create_datasets(args, tokenizer, train_phase=3):
    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    prompt_train_dataset, _ = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_prompt_seq_len)
    if unsupervised_training_enabled:
        unsupervised_train_dataset = get_unsupervised_data(args, tokenizer)
    else:
        unsupervised_train_dataset = None

    # DataLoaders creation:
    data_collator = DataCollatorRLHF(args.max_prompt_seq_len,
                                     args.inference_tp_size)
    if args.local_rank == -1:
        prompt_train_sampler = RandomSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = RandomSampler(
                unsupervised_train_dataset)
    else:
        prompt_train_sampler = DistributedSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = DistributedSampler(
                unsupervised_train_dataset)
    prompt_train_dataloader = DataLoader(
        prompt_train_dataset,
        collate_fn=data_collator,
        sampler=prompt_train_sampler,
        batch_size=args.per_device_train_batch_size)
    if unsupervised_training_enabled:
        unsupervised_train_dataloader = DataLoader(
            unsupervised_train_dataset,
            collate_fn=default_data_collator,
            sampler=unsupervised_train_sampler,
            batch_size=args.per_device_train_batch_size)
    else:
        unsupervised_train_dataloader = [None] * len(
            prompt_train_dataloader)  # basically a dummy dataloader

    num_update_steps_per_epoch = min(len(prompt_train_dataloader), len(unsupervised_train_dataloader)) * \
        (args.per_device_train_batch_size / args.per_device_mini_train_batch_size) * \
        args.ppo_epochs / args.gradient_accumulation_steps
    num_total_iters = int(args.num_train_epochs * num_update_steps_per_epoch)

    return prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    assert not args.offload, "zero-offload is not currently supported but coming soon!"

    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    if unsupervised_training_enabled:
        # if we enable unsupervised training, we need to double the batch size for actor model
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps * 2
    else:
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    # create common tokenizer based on actor model
    tokenizer = AutoTokenizer.from_pretrained(args.actor_model_name_or_path,
                                              fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token

    prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters = create_datasets(
        args=args, tokenizer=tokenizer, train_phase=3)

    # RLHF engine is responsible for creating models, loading checkpoints, ds-initialize models/optims/lr-schedulers
    rlhf_engine = DeepSpeedRLHFEngine(
        actor_model_name_or_path=args.actor_model_name_or_path,
        critic_model_name_or_path=args.critic_model_name_or_path,
        tokenizer=tokenizer,
        num_total_iters=num_total_iters,
        args=args)

    args.end_of_conversation_token = "<|endoftext|>"

    ppo_trainer = DeepSpeedPPOTrainerUnsupervised if unsupervised_training_enabled else DeepSpeedPPOTrainer
    trainer = ppo_trainer(rlhf_engine, args)

    # first number is how many experience-batch to generate, second number is the training batch size, which is the micro-batch size used
    exp_mini_dataset = MiniDataset(args.generation_batch_numbers,
                                   args.per_device_mini_train_batch_size)
    unsup_mini_dataset = MiniDataset(args.generation_batch_numbers,
                                     args.per_device_mini_train_batch_size)

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Generation Batches {min(len(prompt_train_dataloader), len(unsupervised_train_dataloader))}",
            args.global_rank)
        for step, (batch_prompt, batch_unsupervised) in enumerate(
                zip(prompt_train_dataloader, unsupervised_train_dataloader)):
            batch_prompt = to_device(batch_prompt, device)
            if batch_unsupervised is not None:
                batch_unsupervised = to_device(batch_unsupervised, device)
                unsup_dataset = unsup_mini_dataset.add(batch_unsupervised)
            else:
                unsup_dataset = unsup_mini_dataset.add(
                    [[None] * args.per_device_train_batch_size])
            prompts = batch_prompt['prompt']
            length = prompts.size(-1)
            if length > args.max_prompt_seq_len:
                prompts = prompts[:, length - args.max_prompt_seq_len:]
                raise ValueError("Prompt length is too long")

            out = trainer.generate_experience(prompts)
            exp_dataset = exp_mini_dataset.add(out)

            if exp_dataset is not None:
                inner_iter = 0
                critic_loss, actor_loss, unsuper_loss = 0, 0, 0
                average_reward = 0

                if args.actor_gradient_checkpointing:
                    rlhf_engine.actor.gradient_checkpointing_enable()

                for ppo_ep in range(args.ppo_epochs):
                    for i, (exp_data, unsup_data) in enumerate(
                            zip(exp_dataset, unsup_dataset)):
                        actor_loss, critic_loss = trainer.train_rlhf(exp_data)
                        critic_loss += actor_loss.item()
                        actor_loss += critic_loss.item()
                        average_reward += exp_data["rewards"].mean()

                        if unsupervised_training_enabled:
                            unsup_loss = trainer.train_unsupervised(
                                unsup_data, args.unsup_coef)
                            unsuper_loss += unsup_loss.item()

                        inner_iter += 1
                        if args.enable_ema:
                            moving_average(rlhf_engine.actor,
                                           rlhf_engine.actor_ema,
                                           zero_stage=args.actor_zero_stage)

                    random.shuffle(exp_dataset)
                    random.shuffle(unsup_dataset)

                print_rank_0(
                    f'epoch: {epoch}|step: {step}|ppo_ep: {ppo_ep+1}|act_loss: {actor_loss/inner_iter}|cri_loss: {critic_loss/inner_iter}|unsuper_loss: {unsuper_loss/inner_iter}',
                    args.global_rank)
                average_reward = get_all_reduce_mean(average_reward).item()
                print_rank_0(
                    f"average reward score: {average_reward/inner_iter}",
                    args.global_rank)
                print_rank_0(
                    "-------------------------------------------------------------------------------------",
                    args.global_rank)

            if args.actor_gradient_checkpointing:
                rlhf_engine.actor.gradient_checkpointing_disable()

    if args.output_dir is not None:
        print_rank_0('saving model ...')
        rlhf_engine.actor = convert_lora_to_linear_layer(rlhf_engine.actor)
        rlhf_engine.critic = convert_lora_to_linear_layer(rlhf_engine.critic)
        if args.enable_ema:
            rlhf_engine.actor_ema = convert_lora_to_linear_layer(
                rlhf_engine.actor_ema)

        if torch.distributed.get_rank() == 0:
            save_hf_format(rlhf_engine.actor,
                           tokenizer,
                           args,
                           sub_folder='actor')
            save_hf_format(rlhf_engine.critic,
                           tokenizer,
                           args,
                           sub_folder='critic')
            if args.enable_ema:
                save_hf_format(rlhf_engine.actor_ema,
                               tokenizer,
                               args,
                               sub_folder='actor_ema')

        if args.actor_zero_stage == 3:
            save_zero_three_model(rlhf_engine.actor,
                                  global_rank=args.global_rank,
                                  save_dir=os.path.join(
                                      args.output_dir, 'actor'),
                                  zero_stage=args.actor_zero_stage)
            if args.enable_ema:
                save_zero_three_model(rlhf_engine.actor_ema,
                                      global_rank=args.global_rank,
                                      save_dir=os.path.join(
                                          args.output_dir, 'actor_ema'),
                                      zero_stage=args.actor_zero_stage)
        if args.critic_zero_stage == 3:
            save_zero_three_model(rlhf_engine.critic,
                                  global_rank=args.global_rank,
                                  save_dir=os.path.join(
                                      args.output_dir, 'critic'),
                                  zero_stage=args.critic_zero_stage)


if __name__ == "__main__":
    main()
