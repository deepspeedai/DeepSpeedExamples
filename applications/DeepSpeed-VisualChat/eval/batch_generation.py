#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import csv
import sys
from PIL import Image

import torch
import deepspeed
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.utils import print_rank_0, to_device
from utils.model import create_dsvl_model_and_transforms as create_model_and_transforms
import utils.data.DST as DST 




from typing import Iterable
from transformers import AutoTokenizer, set_seed
import json
import collections
import numpy as np 
import random
def load_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

# usage
def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--question-format',
                        type=str,
                        default="prompt_qa",
                        choices=['prompt_qa', 'prompt_choice'],
                        help='question-format')
    parser.add_argument('--question',
                        type=str,
                        default="please describe the image",
                        help='question-format')
    parser.add_argument(
        "--lm_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument("--vision_model_name_or_path", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument(
        "--pretrained_path",
        default=None,
        type=str,
        help="path to pretrained model",
    )
    parser.add_argument(
        "--image_token_length",
        type=int,
        default=256,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        help="path to pretrained model",
    )
    parser.add_argument('--checkpoint_names',
                        nargs='*',
                        default=['runing_check_stage2_v3_epoch10',],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument(
        "--model_name",
        default="dsvl",
        type=str,
        choices=["dsvl", "toy"],
        help="path to pretrained model",
    )
    parser.add_argument(
        "--enable_mmca_attention",
        action='store_true',
        help="enable the new proposed attn, which is similar to cross attention",
    )
    parser.add_argument(
        "--vis_proj",
        type=str,
        default='baseline',
        help="baseline, vit, or perceiver",
    )
    parser.add_argument(
        "--eval_data",
        default="dsvl",
        type=str,
        help="path to eval data",
    )
    parser.add_argument(
        "--output_filename",
        default="results",
        type=str,
        help="path to eval data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="The maximum sequence length.",
    )
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    with open(f'./eval/eval_data/{args.eval_data}.json', 'r') as file:
        data = json.load(file)
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
            
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model_name_or_path,
                                              fast_tokenizer=True)
    tokenizer.padding_side = 'right'
    model, image_processor, tokenizer = create_model_and_transforms(
        text_tokenizer = tokenizer,
        ds_config=None,
        args=args,
    )
    get_results = collections.defaultdict(list)
    for ck_name in args.checkpoint_names:
        ck_path = os.path.join(args.checkpoint_path, ck_name)
        print (ck_path)
        if ck_path is not None:
            model.load_state_dict(torch.load(os.path.join(ck_path, 'pytorch_model.bin'), map_location='cpu'), strict=False) # Z3 wouldn't save pos embeddings (vis and rope)
        else:
            Warning("No checkpoint loaded so you cannot genereate meaningful results")
        #model = model.cuda().half()
        model = model.eval()
        model.projection = model.projection.to('cuda')
        model.vis_encoder = model.vis_encoder.to('cuda')
        model = model.half()
        print_rank_0(model)
        for name in data.keys():
            question_image_list = data[name]
            print (f'{args.eval_data}-------------------------------------{name}')
            images = []
            system_instruct = []
            TEMPLATE = DST.Prompter() # get template
            image_token_dict = DST.get_image_num_map(tokenizer)
            image_num = 0
            for round, q_i_pair in enumerate(question_image_list):
                # print(f'=========round {round+1}==============')
                question = q_i_pair[0]
                if len(q_i_pair) > 1:
                    # seperate by space 
                    image_paths = q_i_pair[1].split(' ')
                    tmp_images = []
                    for image_path in image_paths:
                        image = Image.open(image_path.strip()).convert('RGB')
                        tmp_image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0).cuda().half()
                        tmp_images.append(tmp_image_tensor)                    
                    images = images + tmp_images # get all images
                    with_image = True
                    image_num = len(tmp_images)
                else:
                    image_num = 0
                    with_image = False

                if len(images) > 0:
                    image_tensor = torch.cat(images, dim=0) # cat all images
                else:
                    raise ValueError("No image provided. Did not fix this in the modeling side yet.")

                full_prompt = TEMPLATE(question, with_image=with_image, first_message=(round==0), num_images=image_num)
                full_prompt_ids = tokenizer(full_prompt).input_ids # remove bos token
                if with_image:
                    image_number = len(images)
                    index = full_prompt_ids.index(image_token_dict[DST.DEFAULT_HUMAN_IMAGE_PRETOKEN])
                    full_prompt_ids[index] = image_token_dict[DST.image_mapping_dict[str(image_number)]]
                full_prompt_ids = DST.flatten(full_prompt_ids)
                input_ids = torch.as_tensor([system_instruct + full_prompt_ids]).cuda() # entire input as system instruction for simplicity
                print ('\n',round,question, '||', q_i_pair[-1] )

                generate_output = model.generate(image_tensor, input_ids,
                                                generation_length=256)
                # generation_kwargs={ 'num_beams':2,'num_return_sequences':1,'top_p':1,'do_sample':True, 'temperature':1}
                print('vanilla-->', generate_output[1])
                get_results[name].append([q_i_pair[-1], question, generate_output[1]])
                extend_ids = generate_output[0].cpu().tolist()[0]
                while extend_ids[-1] == tokenizer.pad_token_id:
                    extend_ids.pop()
                while extend_ids[0] == tokenizer.bos_token_id:
                    # llama-2 generates bos token at the beginning
                    extend_ids.pop(0)
                system_instruct = system_instruct + full_prompt_ids + extend_ids # entire input as system instruction for simplicity
                system_instruct = system_instruct + [tokenizer.eos_token_id] # add eos token
                
    with open(f'{args.output_filename}.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['test_name', 'image_path', 'question', 'answer'])
        for test_name, questions in get_results.items():
            for question in questions:
                writer.writerow([test_name] + question)
        
                
        
        
if __name__ == "__main__":
    main()
