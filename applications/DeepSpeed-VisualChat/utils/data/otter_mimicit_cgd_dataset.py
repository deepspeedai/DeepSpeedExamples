# This dataset is from https://huggingface.co/datasets/pufanyi/MIMICIT
import os
import torch
import json
import random
from tqdm import tqdm
from PIL import Image
from .vqa_dataset import VQADataset
from utils.utils import print_rank_0, is_rank_0, get_rank
from .utils import save_debug_image, save_debug_text


class OtterMimicitCgdDataset(VQADataset):
    def __init__(self, data_path, data_debug_path, per_sample_image, tokenizer, vis_processor, **kwargs):
        vis_root = f"{data_path}/coco/train2017"
        assert os.path.isdir(vis_root), f"OtterMimicitCgdDataset image directory {vis_root} not found, you need to download 2017 Train images from https://cocodataset.org/#download"
        ### Below commented code are the images from the MIMIC-IT. We use the original coco images above which are the same and with higher resolution.
        # vis_root = f"{data_path}/MIMIC-IT/CGD_images"
        # if not os.path.isdir(vis_root):
        #     print_rank_0(f"OtterMimicitCgdDataset image directory {vis_root} not found, starting an one-time preprocessing:")
        #     vis_root_file = f"{data_path}/MIMIC-IT/CGD.json"
        #     assert os.path.isfile(vis_root_file), f"OtterMimicitCgdDataset image data {vis_root_file} not found, you need to download it from https://huggingface.co/datasets/pufanyi/MIMICIT"
        #     if is_rank_0():
        #         os.makedirs(vis_root, exist_ok=True)
        #         image_data = json.load(open(vis_root_file, "r"))
        #         image_keys = list(image_data.keys())
        #         for k in tqdm(image_keys):
        #             image = base64.b64decode(image_data[k])
        #             with open(f"{vis_root}/{k}.jpg", 'wb') as f:
        #                 f.write(image)
        # torch.distributed.barrier()

        ann_paths_raw = ["MIMIC-IT/CGD_instructions.json"]
        ann_paths = ["MIMIC-IT/CGD_instructions_merged.json"]
        for idx in range(len(ann_paths)):
            ann_paths_raw[idx] = f"{data_path}/{ann_paths_raw[idx]}"
            ann_paths[idx] = f"{data_path}/{ann_paths[idx]}"
            assert os.path.isfile(ann_paths_raw[idx]), f"OtterMimicitCgdDataset raw annotation file {ann_paths_raw[idx]} not found, you need to download it from https://huggingface.co/datasets/pufanyi/MIMICIT"
            if not os.path.isfile(ann_paths[idx]):
                print_rank_0(f"OtterMimicitCgdDataset annotation file {ann_paths[idx]} not found, starting an one-time preprocessing:")
                if is_rank_0():
                    raw_annotation = json.load(open(ann_paths_raw[idx], "r"))["data"]
                    raw_annotation_keys = list(raw_annotation.keys())
                    random.shuffle(raw_annotation_keys)
                    annotations = []
                    for k in tqdm(raw_annotation_keys):
                        if k in raw_annotation:
                            ann = {}
                            ann["image_ids"] = raw_annotation[k]["image_ids"]
                            ann["instruction"] = [raw_annotation[k]["instruction"]]
                            ann["answer"] = [raw_annotation[k]["answer"]]
                            rel_ins_ids = raw_annotation[k]["rel_ins_ids"]
                            for k_rel in rel_ins_ids:
                                if k_rel in raw_annotation:
                                    ann["instruction"].append(raw_annotation[k_rel]["instruction"])
                                    ann["answer"].append(raw_annotation[k_rel]["answer"])
                                    del raw_annotation[k_rel]
                            annotations.append(ann)
                            del raw_annotation[k]
                    with open(ann_paths[idx], 'w') as f:
                        json.dump(annotations, f)
            torch.distributed.barrier()
        super().__init__(data_path, data_debug_path, per_sample_image, tokenizer, vis_processor,
                         vis_root, ann_paths, **kwargs)

    def _add_instance_ids(self, key="id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def convert_image_id(self, image_id):
        return image_id[8:] + ".jpg"
        # return image_id + ".jpg" ### Change to this if you switch to use images from MIMIC-IT/CGD_images

    def process_image(self, ann, data_debug_path=None, data_debug_counter=0):
        images = ann["image_ids"]
        output_images = []
        for idx in range(len(images)):
            image = images[idx]
            image_path = os.path.join(self.vis_root, self.convert_image_id(image))
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
        question_list = ann["instruction"]
        answer_list = ann["answer"]
        num_convs = len(question_list)
        indexes = list(range(num_convs))
        random.shuffle(indexes)
        conv_list = []
        for conv_id in range(num_convs):
            question = question_list[indexes[conv_id]]
            # remove '<image>' tag and '\n'
            question = question.replace("<image>", "").replace("\n", "")
            answer = answer_list[indexes[conv_id]]
            instruction = self.prompter(question, with_image=(conv_id == 0 and first_message),
                                        first_message=(conv_id == 0 and first_message),
                                        num_images=num_images)
            if conv_id == 0 and first_message:
                instruction = self.post_process_text_image_count(instruction, num_images)
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
