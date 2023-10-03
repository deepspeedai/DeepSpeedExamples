# This file is adapted from https://github.com/open-mmlab/Multimodal-GPT
# This dataset is from https://ocr-vqa.github.io/
import json
import os
import random
import torch

from PIL import Image
from tqdm import tqdm

from .vqa_dataset import VQADataset
from utils.utils import print_rank_0, is_rank_0, get_rank
from .utils import save_debug_image, save_debug_text


class OCRVQADataset(VQADataset):
    def __init__(self, data_path, data_debug_path, per_sample_image, tokenizer, vis_processor,
                 add_eos=True, ignore_instruction=True, **kwargs):
        self.vis_root = f"{data_path}/OCR_VQA/images"
        assert os.path.isdir(self.vis_root), f"OCRVQADataset image directory {self.vis_root} not found, you need to download images from https://ocr-vqa.github.io/"
        ann_paths_raw = ["OCR_VQA/dataset.json"]
        ann_paths = ["OCR_VQA/dataset_processed.json"]
        real_ann_paths = []
        for idx in range(len(ann_paths_raw)):
            ann_path_raw = f"{data_path}/{ann_paths_raw[idx]}"
            assert os.path.isfile(ann_path_raw), f"OCRVQADataset raw annotation file {ann_path_raw} not found, you need to download it from https://ocr-vqa.github.io/"
            ann_path = f"{data_path}/{ann_paths[idx]}"
            real_ann_paths.append(ann_path)
            if not os.path.isfile(ann_path):
                print_rank_0(f"OCRVQADataset annotation file {ann_path_raw} not found, starting an one-time preprocessing:")
                raw_annotation = json.load(open(ann_path_raw, "r"))
                raw_annotation_keys = list(raw_annotation.keys())
                for k in tqdm(raw_annotation_keys):
                    ext=os.path.splitext(raw_annotation[k]['imageURL'])[1]
                    outputFile = '%s%s'%(k,ext)
                    image_path = os.path.join(self.vis_root, outputFile)
                    image = Image.open(image_path).convert("RGB")
                    if image.size[0] > 1 and image.size[1] > 1:
                        raw_annotation[k]["filename"] = outputFile
                    else:
                        del raw_annotation[k]
                if is_rank_0():
                    with open(ann_path, 'w') as f:
                        json.dump(list(raw_annotation.values()), f)
            torch.distributed.barrier()
        super().__init__(data_path, data_debug_path, per_sample_image, tokenizer, vis_processor,
                         self.vis_root, real_ann_paths, **kwargs)

    def process_image(self, ann, data_debug_path=None, data_debug_counter=0):
        image_path = os.path.join(self.vis_root, ann["filename"])
        save_debug_image(image_path, data_debug_path, data_debug_counter, get_rank(), img_idx=0)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        try:
            image = image['pixel_values'][0]
            return image
        except:
            return image

    def process_text(self, ann, data_debug_path=None, data_debug_counter=0, first_message=True):
        index = random.choice(list(range(len(ann["questions"]))))
        question = ann["questions"][index]
        answer = ann["answers"][index]

        instruction = self.prompter(question, with_image=True, first_message=first_message)
        save_debug_text([instruction, answer], data_debug_path, data_debug_counter, get_rank())
        return dict(instruction=instruction, answer=answer)
