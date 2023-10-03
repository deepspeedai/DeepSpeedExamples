# This dataset is from https://llava-vl.github.io/ and https://huggingface.co/datasets/pufanyi/MIMICIT
# This dataset blends llava, llava_dial, and otter_mimicit_cgd datasets, which is possible because
# all of them use coco images. In each sample of LlavaOtterBlendDataset, there will first have at
# least one instruction-answer pair from llava/llava_dial, then followed by at least one
# instruction-answer pair from otter_mimicit_cgd.
import os
import torch
import json
import random
from tqdm import tqdm
from PIL import Image
from .vqa_dataset import VQADataset
from utils.utils import print_rank_0, is_rank_0, get_rank
from .utils import save_debug_image, save_debug_text


class LlavaOtterBlendDataset(VQADataset):
    def __init__(self, data_path, data_debug_path, per_sample_image, followup, tokenizer, vis_processor, **kwargs):
        vis_root = f"{data_path}/coco/train2017"
        assert os.path.isdir(vis_root), f"LlavaOtterBlendDataset image directory {vis_root} not found, you need to download 2017 Train images from https://cocodataset.org/#download"

        otter_mimicit_cgd = f"{data_path}/MIMIC-IT/CGD_instructions.json"
        llava = [f"{data_path}/llava/detail_23k.json", f"{data_path}/llava/complex_reasoning_77k.json", f"{data_path}/llava/conversation_58k.json"]
        ann_path_otter = f"{data_path}/LlavaOtterBlendDataset_instructions_otter.json"
        ann_path_llava = f"{data_path}/LlavaOtterBlendDataset_instructions_llava.json"
        if not os.path.isfile(ann_path_llava):
            print_rank_0(f"LlavaOtterBlendDataset llava annotation file {ann_path_llava} not found, starting an one-time preprocessing:")
            if is_rank_0():
                annotations_llava = {}
                for llava_ann in llava:
                    assert os.path.isfile(llava_ann), f"LlavaOtterBlendDataset raw annotation file {llava_ann} not found, you need to download it from https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K"
                    raw_annotation = json.load(open(llava_ann, "r"))
                    for raw_ann in raw_annotation:
                        if raw_ann["image"] not in annotations_llava:
                            annotations_llava[raw_ann["image"]] = []
                        annotations_llava[raw_ann["image"]].append(raw_ann["conversations"])
                with open(ann_path_llava, 'w') as f:
                    json.dump(annotations_llava, f)
        torch.distributed.barrier()
        self.ann_llava = json.load(open(ann_path_llava, "r"))
        if not os.path.isfile(ann_path_otter):
            print_rank_0(f"LlavaOtterBlendDataset otter annotation file {ann_path_otter} not found, starting an one-time preprocessing:")
            if is_rank_0():
                assert os.path.isfile(otter_mimicit_cgd), f"LlavaOtterBlendDataset raw annotation file {otter_mimicit_cgd} not found, you need to download it from https://huggingface.co/datasets/pufanyi/MIMICIT"
                raw_annotation = json.load(open(otter_mimicit_cgd, "r"))["data"]
                raw_annotation_keys = list(raw_annotation.keys())
                annotations_otter = []
                for k in tqdm(raw_annotation_keys):
                    if k in raw_annotation:
                        ann = {}
                        ann["image_ids"] = [self.convert_image_id(x) for x in raw_annotation[k]["image_ids"]]
                        meet_criteria = True
                        for midx in range(len(ann["image_ids"])-1):
                            if ann["image_ids"][midx] not in self.ann_llava:
                                meet_criteria = False
                        if meet_criteria: # If any image (except the last image) doesn't have llava conversation, we won't be able to build valid sample with correct image order
                            ann["instruction"] = [raw_annotation[k]["instruction"]]
                            ann["answer"] = [raw_annotation[k]["answer"]]
                            rel_ins_ids = raw_annotation[k]["rel_ins_ids"]
                            for k_rel in rel_ins_ids:
                                if k_rel in raw_annotation:
                                    ann["instruction"].append(raw_annotation[k_rel]["instruction"])
                                    ann["answer"].append(raw_annotation[k_rel]["answer"])
                                    del raw_annotation[k_rel]
                            annotations_otter.append(ann)
                        del raw_annotation[k]
                with open(ann_path_otter, 'w') as f:
                    json.dump(annotations_otter, f)
        torch.distributed.barrier()
        super().__init__(data_path, data_debug_path, per_sample_image, tokenizer, vis_processor,
                         vis_root, [ann_path_otter], **kwargs)
        self.followup = followup

    def _add_instance_ids(self, key="id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def convert_image_id(self, image_id):
        return image_id[8:] + ".jpg"

    def process_image(self, ann, data_debug_path=None, data_debug_counter=0):
        images = ann["image_ids"]
        output_images = []
        for idx in range(len(images)):
            image = images[idx]
            image_path = os.path.join(self.vis_root, image)
            save_debug_image(image_path, data_debug_path, data_debug_counter, get_rank(), img_idx=idx)
            image = Image.open(image_path).convert("RGB")

            image = self.vis_processor(image)
            try:
                image = image['pixel_values'][0]
            except:
                image = image
            output_images.append(image)
        
        return output_images

    def process_text(self, ann, data_debug_path=None, data_debug_counter=0, first_message=False, num_images=1):
        images = ann["image_ids"]
        processed_images = {}
        conv_list = []
        # At least one conversation from llava
        for idx in range(len(images)):
            img_key = images[idx]
            if img_key in self.ann_llava:
                conversations = self.ann_llava[img_key]
                min_num_draw = 1 if idx < (len(images) - 1) else 0 # The last image could have 0 llava conversation since it won't break image order
                num_draw = random.randint(min_num_draw, len(conversations))
                chosen = random.sample(list(range(len(conversations))), num_draw)
                for cid in chosen:
                    conv = conversations[cid]
                    num_convs = len(conv) // 2
                    for conv_id in range(num_convs):
                        question = conv[int(2*conv_id)]["value"]
                        # remove '<image>' tag and '\n'
                        with_image = img_key not in processed_images
                        question = question.replace("<image>", "").replace("\n", "")
                        answer = conv[int(2*conv_id+1)]["value"]
                        instruction = self.prompter(question, with_image=with_image, first_message=(len(conv_list) == 0 and first_message))
                        if with_image:
                            instruction = self.post_process_text_image_count(instruction, 1, offset=len(processed_images))
                        single_conv = dict(instruction=instruction, answer=answer)
                        conv_list.append(single_conv)
                        processed_images[img_key] = 1

        # At least one conversation from otter
        question_list = ann["instruction"]
        answer_list = ann["answer"]
        num_convs = len(question_list)
        num_draw = random.randint(1, num_convs)
        chosen = random.sample(list(range(num_convs)), num_draw)
        for cid in chosen:
            question = question_list[cid]
            # remove '<image>' tag and '\n'
            question = question.replace("<image>", "").replace("\n", "")
            answer = answer_list[cid]
            num_images = len(images) - len(processed_images)
            instruction = self.prompter(question, with_image=(num_images > 0),
                                        first_message=(len(conv_list) == 0),
                                        num_images=num_images)
            if num_images > 0:
                instruction = self.post_process_text_image_count(instruction, num_images, offset=len(processed_images))
            single_conv = dict(instruction=instruction, answer=answer)
            conv_list.append(single_conv)
            processed_images = images
        # Follow-up llava conversations
        if self.followup:
            image_tags = {0: ["In image 1, ", "In image a, ", "In the first image, "], 1: ["In image 2, ", "In image b, ", "In the second image, "]}
            for idx in range(len(images)):
                img_key = images[idx]
                if img_key in self.ann_llava:
                    conversations = self.ann_llava[img_key]
                    # min_num_draw = 1
                    # num_draw = random.randint(min_num_draw, len(conversations))
                    num_draw = 1 # To avoid making too complex conversation, we limit num of follow-up conversation to 1 per image
                    chosen = random.sample(list(range(len(conversations))), num_draw)
                    for cid in chosen:
                        conv = conversations[cid]
                        num_convs = len(conv) // 2
                        for conv_id in range(num_convs):
                            question = conv[int(2*conv_id)]["value"]
                            # remove '<image>' tag and '\n'
                            question = question.replace("<image>", "").replace("\n", "")
                            answer = conv[int(2*conv_id+1)]["value"]
                            # Add image tags so the model knows which image we are referring
                            chosen_tag = random.choice(image_tags[idx])
                            question = chosen_tag + question[0].lower() + question[1:]
                            answer = chosen_tag + answer[0].lower() + answer[1:]
                            instruction = self.prompter(question, with_image=False, first_message=False)
                            single_conv = dict(instruction=instruction, answer=answer)
                            conv_list.append(single_conv)
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
