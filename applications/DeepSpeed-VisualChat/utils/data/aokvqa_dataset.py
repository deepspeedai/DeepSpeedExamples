# This file is adapted from https://github.com/open-mmlab/Multimodal-GPT
# This dataset is from https://allenai.org/project/a-okvqa/home
import os
import random
from PIL import Image

from .vqa_dataset import VQADataset
from utils.utils import get_rank
from .utils import save_debug_image, save_debug_text

REASON_QUESTIONS = [
    "Why?",
    "Why is this?",
    "And why?",
    "What is the reason?",
    "And can you tell me why?",
    "Can you tell me why?",
    "Can you tell me the reason?",
]


class AOKVQADataset(VQADataset):
    def __init__(self, data_path, data_debug_path, per_sample_image, tokenizer, vis_processor, **kwargs):
        vis_root = f"{data_path}/coco/train2017"
        assert os.path.isdir(vis_root), f"AOKVQADataset image directory {vis_root} not found, you need to download 2017 Train images from https://cocodataset.org/#download"
        ann_paths = ["aokvqa/annotations/aokvqa_v1p0_train.json"]
        for idx in range(len(ann_paths)):
            ann_paths[idx] = f"{data_path}/{ann_paths[idx]}"
            assert os.path.isfile(ann_paths[idx]), f"AOKVQADataset annotation file {ann_paths[idx]} not found, you need to download it from https://allenai.org/project/a-okvqa/home"
        super().__init__(data_path, data_debug_path, per_sample_image, tokenizer, vis_processor,
                         vis_root, ann_paths, **kwargs)

    def process_text(self, ann, data_debug_path=None, data_debug_counter=0, first_message=True):
        question = ann["question"]
        question = question + " " + random.choice(REASON_QUESTIONS)

        choices = ann["choices"]
        true_answer = choices[ann["correct_choice_idx"]]
        answer = "The answer is " + true_answer + ". Because " + " ".join(ann["rationales"])

        is_option = random.random() < self.option_prob and len(choices) > 1 # let's not do option for now
        # if is_option:
        #     instruction = self.prompter(question, choices)
        # else:
        instruction = self.prompter(question, with_image=True, first_message=first_message)
        save_debug_text([instruction, answer], data_debug_path, data_debug_counter, get_rank())
        return dict(instruction=instruction, answer=answer)
    
    def process_image(self, ann, data_debug_path=None, data_debug_counter=0):
        image_path = os.path.join(self.vis_root, str(ann["image_id"]).rjust(12, '0') + ".jpg")
        save_debug_image(image_path, data_debug_path, data_debug_counter, get_rank(), img_idx=0)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        try:
            image = image['pixel_values'][0]
            return image
        except:
            return image
