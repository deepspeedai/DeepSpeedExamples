# This file is adapted from https://github.com/open-mmlab/Multimodal-GPT
# This dataset is from https://llava-vl.github.io/
import os
from .vqa_dataset import VQADataset
from utils.utils import get_rank
from .utils import save_debug_text


class LlavaDataset(VQADataset):
    def __init__(self, data_path, data_debug_path, per_sample_image, tokenizer, vis_processor, **kwargs):
        vis_root = f"{data_path}/coco/train2017"
        assert os.path.isdir(vis_root), f"LlavaDataset image directory {vis_root} not found, you need to download 2017 Train images from https://cocodataset.org/#download"
        ann_paths = ["llava/detail_23k.json", "llava/complex_reasoning_77k.json"]
        for idx in range(len(ann_paths)):
            ann_paths[idx] = f"{data_path}/{ann_paths[idx]}"
            assert os.path.isfile(ann_paths[idx]), f"LlavaDataset annotation file {ann_paths[idx]} not found, you need to download it from https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K"
        super().__init__(data_path, data_debug_path, per_sample_image, tokenizer, vis_processor,
                         vis_root, ann_paths, **kwargs)

    def _add_instance_ids(self, key="id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def process_text(self, ann, data_debug_path=None, data_debug_counter=0, first_message=False):
        question = ann["conversations"][0]["value"]
        # remove '<image>' tag and '\n'
        question = question.replace("<image>", "").replace("\n", "")
        answer = ann["conversations"][1]["value"]
        instruction = self.prompter(question, with_image=True, first_message=first_message)
        save_debug_text([instruction, answer], data_debug_path, data_debug_counter, get_rank())
        return dict(instruction=instruction, answer=answer)
