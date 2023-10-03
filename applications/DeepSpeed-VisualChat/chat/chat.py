#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import sys
from PIL import Image

import torch
import deepspeed

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.utils import print_rank_0
from utils.model import create_dsvl_model_and_transforms
import utils.data.DST as DST 
from transformers import AutoTokenizer
from termcolor import colored
import re

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "CLI chat")
    parser.add_argument(
        "--lm_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument("--vision_model_name_or_path", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        help="path to pretrained model",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=4096,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--generation_length_per_round",
        type=int,
        default=256,
        help="The generation length per conversation round.",
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
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def get_user_text_input():
    tmp = input(colored("Enter input (type 'quit' to exit, 'clear' to clean memory): ", 'green'))
    return tmp, tmp == "quit", tmp == "clear"

def get_user_image_input():
    tmp = input(colored("Enter image pathes, seperate by space (only support one image per time for now) (type 'na' for empty image): ", 'blue'))
    return tmp, not tmp == "na"

def main():
    args = parse_args()    
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model_name_or_path,
                                              fast_tokenizer=True)
    tokenizer.padding_side = 'right'
    model, image_processor, tokenizer = create_dsvl_model_and_transforms(
        text_tokenizer = tokenizer,
        ds_config=None,
        args=args,
    )

    model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, 'pytorch_model.bin'), map_location='cpu'), strict=False) # Z3 wouldn't save pos embeddings (vis and rope)
    
    model = model.eval()
    model.projection = model.projection.to('cuda')
    model.vis_encoder = model.vis_encoder.to('cuda')
    model = model.half()
    print_rank_0(model) 
    
    num_rounds  = 0 
    images = []
    system_instruct = []
    TEMPLATE = DST.Prompter() # get template
    image_num_token_list = [DST.IMAGE_NUM_1, DST.IMAGE_NUM_2, DST.IMAGE_NUM_3, DST.IMAGE_NUM_4, DST.IMAGE_NUM_5, DST.IMAGE_NUM_6, DST.IMAGE_NUM_7, DST.IMAGE_NUM_8]
    
    while True:
        num_rounds  += 1
        while True:
            # it is super easy to make mistake here, so we need to be careful
            image_input, with_image = get_user_image_input()
            if with_image:
                try:
                    # seperate by space 
                    image_paths = image_input.split(' ')
                    tmp_images = []
                    for image_path in image_paths:
                        image = Image.open(image_path).convert('RGB')
                        tmp_image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0).cuda().half()
                        tmp_images.append(tmp_image_tensor) # in case the last image path is wrong
                except:
                    print(colored("Invalid image path, please try again", 'red'))
                    continue
                if len(images) + len(tmp_images) > 8:
                    print(colored("Too many images, we at most support 8 images. please try again", 'red'))
                    continue
                images = images + tmp_images # get all images
                image_num = len(tmp_images)
                break
            else:
                image_num = 0
                break
        assert len(images) >= 1, "We need at least one image to begin the conversation for now."
        if len(images) > 0:
            image_tensor = torch.cat(images, dim=0) # cat all images
        else:
            image_tensor = None

        text_input, quit, clear = get_user_text_input()
        if quit:
            break
        if clear:
            num_rounds = 0 
            images = []
            system_instruct = []
            image_num_token_list = [DST.IMAGE_NUM_1, DST.IMAGE_NUM_2, DST.IMAGE_NUM_3, DST.IMAGE_NUM_4, DST.IMAGE_NUM_5, DST.IMAGE_NUM_6, DST.IMAGE_NUM_7, DST.IMAGE_NUM_8]
            continue
        

        full_prompt = TEMPLATE(text_input, with_image=with_image, first_message=(num_rounds==1), num_images=image_num)
        if with_image:
            for i in range(image_num):
                full_prompt = re.sub(DST.DEFAULT_HUMAN_IMAGE_PRETOKEN, image_num_token_list.pop(0), full_prompt, count=1)
                    

        full_prompt_ids = tokenizer(full_prompt).input_ids # remove bos token
        
        input_ids = torch.as_tensor([system_instruct + full_prompt_ids]).cuda() # entire input as system instruction for simplicity
        generate_output = model.generate(image_tensor, input_ids, generation_length=args.generation_length_per_round)
        extend_ids = generate_output[0].cpu().tolist()[0]
        while extend_ids[-1] == tokenizer.pad_token_id:
            extend_ids.pop()
        while extend_ids[0] == tokenizer.bos_token_id:
            extend_ids.pop(0)
        system_instruct = system_instruct + full_prompt_ids + extend_ids # entire input as system instruction for simplicity
        system_instruct = system_instruct + [tokenizer.eos_token_id] # add eos token

        print(f"=========== Round {num_rounds} ===========")
        print(tokenizer.decode(system_instruct))
        
        
if __name__ == "__main__":
    main()
