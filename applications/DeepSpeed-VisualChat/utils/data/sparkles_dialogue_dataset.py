# This dataset is from https://github.com/HYPJUDY/Sparkles
import os
import torch
import json
import random
import re
from PIL import Image
from .vqa_dataset import VQADataset
from utils.utils import print_rank_0, is_rank_0, get_rank
from .utils import save_debug_image, save_debug_text


class SparklesDialogueDataset(VQADataset):
    def __init__(self, data_path, data_debug_path, per_sample_image, tokenizer, vis_processor, **kwargs):
        vis_root = ["SparklesDialogueCC/images", "SparklesDialogueVG/images"]
        for idx in range(len(vis_root)):
            vis_root[idx] = f"{data_path}/{vis_root[idx]}"
            assert os.path.isdir(vis_root[idx]), f"SparklesDialogueDataset image directory {vis_root[idx]} not found, you need to download it from https://github.com/HYPJUDY/Sparkles"

        ann_path_raw = ["SparklesDialogueCC/annotations/SparklesDialogueCC.json",
                        "SparklesDialogueVG/annotations/SparklesDialogueVG.json"]
        for idx in range(len(ann_path_raw)):
            ann_path_raw[idx] = f"{data_path}/{ann_path_raw[idx]}"
            assert os.path.isfile(ann_path_raw[idx]), f"SparklesDialogueDataset annotation file {ann_path_raw[idx]} not found, you need to download it from https://github.com/HYPJUDY/Sparkles"
        ann_path = f"{data_path}/SparklesDialogue.json"
        
        if not os.path.isfile(ann_path):
            print_rank_0(f"SparklesDialogueDataset: starting an one-time preprocessing:")
            if is_rank_0():
                annotations = []
                for a_idx in range(len(ann_path_raw)):
                    raw_annotation = json.load(open(ann_path_raw[a_idx], "r"))
                    for raw_ann in raw_annotation:
                        meet_criteria = True
                        if len(raw_ann["dialogue"]) % 2 != 0:
                            meet_criteria = False
                        raw_ann["image_path"] = vis_root[a_idx]
                        num_img = 0
                        for d_idx in range(len(raw_ann["dialogue"])):
                            if d_idx % 2 == 0 and raw_ann["dialogue"][d_idx]["role"] != "user":
                                meet_criteria = False
                            if d_idx % 2 == 1 and raw_ann["dialogue"][d_idx]["role"] != "assistant":
                                meet_criteria = False
                            if "images" in raw_ann["dialogue"][d_idx]:
                                for img in raw_ann["dialogue"][d_idx]["images"]:
                                    img_id = img["image_id"]
                                    num_img += 1
                                    if not os.path.isfile(f"{vis_root[a_idx]}/{img_id}.jpg"):
                                        meet_criteria = False
                        if num_img > 8: # Currently only use conversations with <= 8 images
                            meet_criteria = False
                        if meet_criteria:
                            annotations.append(raw_ann)
                with open(ann_path, 'w') as f:
                    json.dump(annotations, f)
            torch.distributed.barrier()
        super().__init__(data_path, data_debug_path, per_sample_image, tokenizer, vis_processor,
                         vis_root, [ann_path], **kwargs)
        self.image_tag_dict = [{0: "image a", 1: "image b", 2: "image c", 3: "image d", 4: "image e", 5: "image f", 6: "image g", 7: "image h"},
                               {0: "image A", 1: "image B", 2: "image C", 3: "image D", 4: "image E", 5: "image F", 6: "image G", 7: "image H"},
                               {0: "the first image", 1: "the second image", 2: "the third image", 3: "the fourth image",
                                4: "the fifth image", 5: "the sixth image", 6: "the seventh image", 7: "the eighth image"}]

    def _add_instance_ids(self, key="id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def process_image(self, ann, data_debug_path=None, data_debug_counter=0):
        output_images = []
        img_counter = 0
        for dialogue in ann["dialogue"]:
            if "images" in dialogue:
                for img in dialogue["images"]:
                    image_path = os.path.join(ann["image_path"], str(img["image_id"]) + ".jpg")
                    save_debug_image(image_path, data_debug_path, data_debug_counter,
                                     get_rank(), img_idx=img_counter)
                    img_counter += 1
                    image = Image.open(image_path).convert("RGB")

                    image = self.vis_processor(image)
                    try:
                        image = image['pixel_values'][0]
                    except:
                        image = image
                    output_images.append(image)
        
        return output_images

    def process_text(self, ann, data_debug_path=None, data_debug_counter=0, first_message=False, num_images=1):
        tag_dict = random.choice(self.image_tag_dict)
        regex = re.compile(r'((?<=[\.\?!]\s)(\w+)|(^\w+))')
        def capitalize_sentence(match):
            return(match.group().capitalize())
        to_replace = []
        conv_list = []
        num_convs = len(ann["dialogue"]) // 2
        tot_num_image = 0
        for conv_id in range(num_convs):
            with_image = False
            num_image = 0
            if "images" in ann["dialogue"][int(2*conv_id)]:
                with_image = True
                for img in ann["dialogue"][int(2*conv_id)]["images"]:
                    img_id = img["image_id"]
                    tag_replace = [f"IMAGE#{img_id}", tag_dict[len(to_replace)]]
                    to_replace.append(tag_replace)
                    num_image += 1
            question = ann["dialogue"][int(2*conv_id)]["content"]
            # remove '<Img>' tag and '\n'
            question = question.replace("<Img><ImageHere></Img>", "").replace("\n", "")
            answer = ann["dialogue"][int(2*conv_id+1)]["content"]
            for idx in range(len(to_replace)):
                question = question.replace(to_replace[idx][0], f"%temp{idx}%")
                answer = answer.replace(to_replace[idx][0], f"%temp{idx}%")
            for idx in range(len(to_replace)):
                question = question.replace(f"%temp{idx}%", to_replace[idx][1])
                answer = answer.replace(f"%temp{idx}%", to_replace[idx][1])
            question = regex.sub(capitalize_sentence, question)
            answer = regex.sub(capitalize_sentence, answer)
            instruction = self.prompter(question, with_image=with_image, first_message=(len(conv_list) == 0 and first_message), num_images=num_image)
            if with_image:
                instruction = self.post_process_text_image_count(instruction, num_image, offset=tot_num_image)
            single_conv = dict(instruction=instruction, answer=answer)
            conv_list.append(single_conv)
            tot_num_image += num_image

        save_debug_text(conv_list, data_debug_path, data_debug_counter, get_rank())
        return conv_list

    def __getitem__(self, index):
        ann = self.annotation[index][0] # self.annotation[index] is a list because of "self.annotation = DST.random_grouping(self.annotation, self.per_sample_image)" in VQADataset init
        images_list = self.process_image(ann,
                                    data_debug_path=self.data_debug_path,
                                    data_debug_counter=self.data_debug_counter)
        text_list = self.process_text(ann,
                                    data_debug_path=self.data_debug_path,
                                    data_debug_counter=self.data_debug_counter,
                                    first_message=True,
                                    num_images=len(images_list))

        self.data_debug_counter += 1
        res_list = []
        for text in text_list:
            single_res = self.tokenize(text)
            res_list.append(single_res)

        input_ids = []
        attention_mask = []
        labels = []
        for res in res_list:
            input_ids.extend(res["input_ids"])
            attention_mask.extend(res["attention_mask"])
            labels.extend(res["labels"])
        
        res = dict(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        res.update(image=images_list)
        res.update(image_num=len(images_list))

        return res
