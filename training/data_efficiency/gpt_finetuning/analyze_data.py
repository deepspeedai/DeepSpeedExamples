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

from deepspeed.runtime.data_pipeline.data_sampling.data_analyzer \
    import DataAnalyzer
from deepspeed.runtime.data_pipeline.data_sampling.indexed_dataset \
    import MMapIndexedDataset

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
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
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
    parser.add_argument('--analyzing_task', type=str, required=True,
                       default=None,
                       choices=['map',
                                'reduce'],
                       help='What type of analyzing task to perform.')
    parser.add_argument('--analyzing_metric', type=str, nargs='+', default=[],
                       help='What kinds of metrics to analyze.')
    parser.add_argument('--analyzing_num_workers', type=int, default=1,
                       help='Number of workers. Each worker could be a single CPU node.')
    parser.add_argument('--analyzing_worker_id', type=int, default=0,
                       help='Worker id of current node.')
    parser.add_argument('--analyzing_num_threads', type=int, default=1,
                       help='Number of threads for each worker.')
    parser.add_argument('--analyzing_num_threads_reduce', type=int, default=1,
                       help='Number of threads for each worker.')
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

    return args


def main():
    args = parse_args()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

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

    def add_index(x, indice):
        x['index'] = [indice]
        return x

    lm_datasets["train"] = lm_datasets["train"].map(add_index, with_indices=True)

    train_ds = lm_datasets["train"]

    def metric_total_vocab_freq(data):
        torch_data = data['input_ids']
        frequency = torch.bincount(torch_data.view(-1),
            minlength=len(tokenizer)+1)
        return frequency

    def metric_vocab_rarity(data):
        rarity = []
        torch_data = data['input_ids']
        # Do one by one to avoid too high memory consumption
        for row in range(torch_data.size()[0]):
            rarity.append(int(torch.sum(args.total_vocab_freq[torch_data[row]]).item()))
        rarity = torch.tensor(rarity, dtype=torch.long)
        print(f"rarity min {min(rarity)}, max {max(rarity)}, len {len(rarity)}, avg {sum(rarity)/len(rarity)}")
        return rarity

    def get_metric_function(metric_name):
        if metric_name == 'total_vocab_freq':
            return metric_total_vocab_freq
        if metric_name == 'vocab_rarity':
            return metric_vocab_rarity

    def get_metric_type(metric_name):
        if metric_name == 'total_vocab_freq':
            return 'accumulate_value_over_samples'
        if metric_name == 'vocab_rarity':
            return 'single_value_per_sample'

    def run_map():
        if 'vocab_rarity' in args.analyzing_metric or 'seqlen_vocab_rarity' in args.analyzing_metric:
            total_vocab_freq_fname = f"{args.output_dir}/total_vocab_freq/total_vocab_freq_metric_value"
            assert os.path.isfile(f"{total_vocab_freq_fname}.bin") and os.path.isfile(f"{total_vocab_freq_fname}.idx"), "To analyze vocab rarity, first need to analyze the total vocab freq."
            total_vocab_freq = MMapIndexedDataset(total_vocab_freq_fname, skip_warmup=True)
            total_vocab_freq = np.copy(total_vocab_freq[0])
            total_vocab_freq[total_vocab_freq == 0] = 1 # Avoid log(0) error
            total_vocab_freq = np.log(total_vocab_freq/sum(total_vocab_freq)) * -1
            args.total_vocab_freq = torch.tensor(total_vocab_freq, dtype=torch.double)
        metric_functions = [get_metric_function(x) for x in args.analyzing_metric]
        metric_types = [get_metric_type(x) for x in args.analyzing_metric]
        # For metric_dtypes we int64 by default since it could be hard to estimate
        # the appropriate dtype before the mapping analysis. During reduce where
        # we merge the analysis results, the DataAnalyzer will automatically choose
        # the dtype of merged result file as the smallest one that meet the range
        # requirement.
        metric_dtypes = [np.int64 for x in args.analyzing_metric]
        start = time.time()
        data_analyzer = DataAnalyzer(train_ds,
            num_workers=args.analyzing_num_workers,
            worker_id=args.analyzing_worker_id,
            num_threads=args.analyzing_num_threads,
            collate_fn=default_data_collator,
            batch_size=args.per_device_train_batch_size, metric_names=args.analyzing_metric,
            metric_functions=metric_functions, metric_types=metric_types,
            metric_dtypes=metric_dtypes, save_path=args.output_dir)
        data_analyzer.run_map()
        duration = (time.time() - start) / 3600.0
        print(f"map job finished in {duration} hr.")

    def run_reduce():
        metric_functions = [get_metric_function(x) for x in args.analyzing_metric]
        metric_types = [get_metric_type(x) for x in args.analyzing_metric]
        metric_dtypes = [np.int64 for x in args.analyzing_metric]
        start = time.time()
        data_analyzer = DataAnalyzer(train_ds,
            num_workers=args.analyzing_num_workers,
            num_threads=args.analyzing_num_threads,
            num_threads_reduce=args.analyzing_num_threads_reduce,
            collate_fn=default_data_collator,
            batch_size=args.per_device_train_batch_size, metric_names=args.analyzing_metric,
            metric_functions=metric_functions, metric_types=metric_types,
            metric_dtypes=metric_dtypes, save_path=args.output_dir)
        data_analyzer.run_reduce()
        duration = (time.time() - start) / 3600.0
        print(f"reduce job finished in {duration} hr.")
    
    if args.analyzing_task == 'map':
        run_map()
    elif args.analyzing_task == 'reduce':
        run_reduce()
    
if __name__ == "__main__":
    main()
