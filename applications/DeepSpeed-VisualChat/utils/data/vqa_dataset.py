# This file is adapted from https://github.com/open-mmlab/Multimodal-GPT

"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import copy
import json
import os
import random
from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from transformers import LlamaTokenizer
import utils.data.DST as DST 
from utils.utils import get_rank
from .utils import save_debug_image, save_debug_text
import re

class VQADataset(Dataset):
    def __init__(
        self,
        data_path,
        data_debug_path,
        per_sample_image,
        tokenizer,
        vis_processor=None,
        vis_root=None,
        ann_paths=[],
        add_eos=True,
        ignore_instruction=True,
        sample_image=False,
        annotation_key=None
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        if hasattr(tokenizer, "add_eos_token"):
            assert tokenizer.add_eos_token is False, "tokenizer should not add eos token by default"
        self.tokenizer: LlamaTokenizer = tokenizer
        self.data_path = data_path
        self.data_debug_path = data_debug_path
        self.data_debug_counter = 0
        self.vis_root = vis_root
        self.per_sample_image = per_sample_image
        print('check tokenizer',  self.tokenizer)
        self.annotation = []
        for ann_path in ann_paths:
            if annotation_key is None:
                self.annotation.extend(json.load(open(ann_path, "r")))
            else:
                self.annotation.extend(json.load(open(ann_path, "r"))[annotation_key])
        self.sample_image = sample_image
        if self.sample_image:
            print("randomly sample one annotation for each image") 
            self.annotation = self.parse_annotation(self.annotation)

        self.annotation = DST.random_grouping(self.annotation, self.per_sample_image)

        self.vis_processor = vis_processor

        self.option_prob = 0.5
        self.prompter = DST.Prompter()
        self.add_eos = add_eos
        self.ignore_instruction = ignore_instruction
        self.system_instruct = None
        self.image_token_dict = DST.get_image_num_map(self.tokenizer)
        self.cat_number()

    def parse_annotation(self, annotation):
        image_list = defaultdict(list)
        for ann in annotation:
            image_list[ann["image"]].append(ann)
            
        annotation = []
        for ann_list in image_list.values():
            annotation.append(random.choice(ann_list))
        
        return annotation

    def __len__(self):
        return len(self.annotation)

    def cat_number(self):
        tmp = len(self.annotation) // self.per_sample_image
        self.arithmetic_progression_multi_image = [tmp * i for i in range(self.per_sample_image)]

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def process_image(self, ann, data_debug_path=None, data_debug_counter=0):
        image_path = os.path.join(self.vis_root, ann["image"])
        save_debug_image(image_path, data_debug_path, data_debug_counter, get_rank(), img_idx=0)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        try:
            image = image['pixel_values'][0]
            return image
        except:
            return image
    
    def post_process_text_image_count(self, text, image_num, offset=0):
        for i in range(1+offset, image_num+1+offset):
            text = re.sub(DST.DEFAULT_HUMAN_IMAGE_PRETOKEN, DST.image_mapping_dict[f"{i}"], text, count=1)
        return text

    def process_text(self, ann, data_debug_path=None, data_debug_counter=0, first_message=False):
        question = ann["question"]

        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        # create instruction
        true_answer = answers[np.argmax(weights)]
        is_option = random.random() < self.option_prob and len(answers) > 1
        if is_option:
            instruction = self.prompter(question, answers)
        else:
            instruction = self.prompter(question, with_image=True, first_message=first_message)
        save_debug_text([instruction, true_answer], data_debug_path, data_debug_counter, get_rank())
        return dict(instruction=instruction, answer=true_answer)

    def tokenize(self, text):
        res = self.tokenizer(
            text["instruction"] + text["answer"],
            return_tensors=None,
            padding="do_not_pad",
            truncation=True,
            max_length=512,
        )
        if res["input_ids"][-1] != self.tokenizer.eos_token_id and self.add_eos:
            res["input_ids"].append(self.tokenizer.eos_token_id)
            res["attention_mask"].append(1)

        labels = copy.deepcopy(res["input_ids"])
        # ignore instruction_token
        if self.ignore_instruction:
            instruction_token = self.tokenizer(
                text["instruction"], return_tensors=None, padding="do_not_pad", truncation=True, max_length=512
            )
            labels = [DST.DEFAULT_LABEL_PADDING_NUM] * len(instruction_token["input_ids"]) + labels[len(instruction_token["input_ids"]) :]

        res.update(labels=labels)
        return res


    def create_system_instruct(self):
        system_instruct = self.tokenizer(
            DST.DEFAULT_PROMPT,
            return_tensors=None,
            padding="do_not_pad",
            truncation=False,
        )
        # create the system instruction
        self.system_instruct = {
            "input_ids": system_instruct["input_ids"] + [self.tokenizer.eos_token_id],
            "attention_mask": system_instruct["attention_mask"] + [1],
            "labels": (len(system_instruct["input_ids"]) + 1) * [DST.DEFAULT_LABEL_PADDING_NUM],
        }

    def merge_all_images(self, res_list):
        def find_index_and_replace(input_list, attention_mask_list, labels_list, image_number):
            # replace a single number with a list of numbers
            index = input_list.index(self.image_token_dict[DST.DEFAULT_HUMAN_IMAGE_PRETOKEN])
            input_list[index] = self.image_token_dict[DST.image_mapping_dict[str(image_number)]]
            attention_mask_list[index] = [1] * len(self.image_token_dict[DST.image_mapping_dict[str(image_number)]])
            labels_list[index] = [DST.DEFAULT_LABEL_PADDING_NUM] * len(self.image_token_dict[DST.image_mapping_dict[str(image_number)]])
            # flatten nested list
            input_list = DST.flatten(input_list)
            attention_mask_list = DST.flatten(attention_mask_list)
            labels_list = DST.flatten(labels_list)
            return input_list, attention_mask_list, labels_list
        image_number = 0 
        original_output = {"input_ids": [], "attention_mask": [], "labels": [], "image": []} #copy.deepcopy(self.system_instruct)
        # original_output["image"] = []
        for res in res_list:
            # need to check if it has image or not
            if self.image_token_dict[DST.DEFAULT_HUMAN_IMAGE_PRETOKEN] in res["input_ids"]:
                image_number += 1
                res["input_ids"], res["attention_mask"], res["labels"] = find_index_and_replace(res["input_ids"], res["attention_mask"], res["labels"], image_number)
                original_output["image"] = original_output["image"] + [res["image"]]
                # cat res to original_output 
            original_output["input_ids"] = original_output["input_ids"] + res["input_ids"]
            original_output["attention_mask"] = original_output["attention_mask"] + res["attention_mask"]
            original_output["labels"] = original_output["labels"] + res["labels"]
        if image_number == 0:
            raise ValueError("image number should not be zero, we now did not support no-image case.")
        original_output["image_num"] = image_number
        return original_output

    def __getitem__(self, index):
        res_list = []
        for ann in self.annotation[index]:
            image = self.process_image(ann,
                                    data_debug_path=self.data_debug_path,
                                    data_debug_counter=self.data_debug_counter)
            text = self.process_text(ann,
                                    data_debug_path=self.data_debug_path,
                                    data_debug_counter=self.data_debug_counter,
                                    first_message=(not res_list))
            self.data_debug_counter += 1
            res = self.tokenize(text)
            res.update(image=image)
            res.update(text)
            res_list.append(res)
        
        output = self.merge_all_images(res_list)
        return output

    def collater(self, samples):
        image_list, question_list, answer_list, input_id_list, attention_mask_list, labels_list = [], [], [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["instruction"])
            answer_list.append(sample["answer"])
            input_id_list.append(sample["input_ids"])
            attention_mask_list.append(sample["attention_mask"])
            labels_list.append(sample["labels"])

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        max_label_length = max(len(l) for l in labels_list)
        padding_side = self.tokenizer.padding_side
        padded_labels = []
        for l in labels_list:
            remainder = [DST.DEFAULT_LABEL_PADDING_NUM] * (max_label_length - len(l))
            if isinstance(l, list):
                l = l + remainder if padding_side == "right" else remainder + l
            elif padding_side == "right":
                l = np.concatenate([l, remainder]).astype(np.int64)
            else:
                l = np.concatenate([remainder, l]).astype(np.int64)
            padded_labels.append(l)

        padded_samples = self.tokenizer.pad(
            {"input_ids": input_id_list, "attention_mask": attention_mask_list, "labels": padded_labels},
            return_tensors="pt",
            padding="longest",
        )

        # remove all image related tokens
        labels = padded_samples["labels"]
        labels[labels == self.tokenizer.pad_token_id] = DST.DEFAULT_LABEL_PADDING_NUM
        labels[:, 0] = DST.DEFAULT_LABEL_PADDING_NUM
        for k, v in self.image_token_dict.items():
            labels[labels == v] = DST.DEFAULT_LABEL_PADDING_NUM
        return {
            "image": torch.stack(image_list, dim=0),
            "input_ids": padded_samples["input_ids"],
            "attention_mask": padded_samples["attention_mask"],
            "labels": labels,
            "instruction": question_list,
            "answer": answer_list,
        }


class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        # TODO For now only supports datasets with same underlying collater implementations

        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)
