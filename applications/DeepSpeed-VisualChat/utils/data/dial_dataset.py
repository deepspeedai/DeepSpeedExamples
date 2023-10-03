# This file is adapted from https://github.com/open-mmlab/Multimodal-GPT
# This dataset is from https://llava-vl.github.io/
import os
from .vqa_dataset import VQADataset
import utils.data.DST as DST 
from utils.utils import get_rank
from .utils import save_debug_text

class DialDataset(VQADataset):
    def __init__(self, dataset_name, data_path, data_debug_path, per_sample_image, tokenizer, vis_processor, **kwargs):
        if dataset_name == "llava_dial":
            vis_root = f"{data_path}/coco/train2017"
            assert os.path.isdir(vis_root), f"llava_dial image directory {vis_root} not found, you need to download 2017 Train images from https://cocodataset.org/#download"
            ann_paths = ["llava/conversation_58k.json"]
            for idx in range(len(ann_paths)):
                ann_paths[idx] = f"{data_path}/{ann_paths[idx]}"
                assert os.path.isfile(ann_paths[idx]), f"llava_dial annotation file {ann_paths[idx]} not found, you need to download it from https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K"
        super(DialDataset, self).__init__(data_path, data_debug_path, per_sample_image, 
                                          tokenizer, vis_processor, vis_root,
                                          ann_paths, **kwargs)
        self.prompter = DST.Prompter()

    def _add_instance_ids(self, key="id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def process_text(self, anns, data_debug_path=None, data_debug_counter=0, first_message=False):
        num_convs = len(anns["conversations"]) // 2
        conv_list = []
        for conv_id in range(num_convs):
            question = anns["conversations"][int(2*conv_id)]["value"]
            # remove '<image>' tag and '\n'
            with_image = "<image>" in question
            question = question.replace("<image>", "").replace("\n", "")
            answer = anns["conversations"][int(2*conv_id+1)]["value"]
            instruction = self.prompter(question, with_image=with_image, first_message=(conv_id == 0 and first_message))
            single_conv = dict(instruction=instruction, answer=answer)
            conv_list.append(single_conv)
        save_debug_text(conv_list, data_debug_path, data_debug_counter, get_rank())
        return conv_list

    def __getitem__(self, index):
        full_res_list = []
        for ann in self.annotation[index]:
            image = self.process_image(ann,
                                    data_debug_path=self.data_debug_path,
                                    data_debug_counter=self.data_debug_counter)
            text_list = self.process_text(ann,
                                        data_debug_path=self.data_debug_path,
                                        data_debug_counter=self.data_debug_counter,
                                        first_message=(not full_res_list))
            self.data_debug_counter += 1
            res_list = []
            for text in text_list:
                single_res = self.tokenize(text)
                single_res["instruction"] = text["instruction"]
                single_res["answer"] = text["answer"]
                res_list.append(single_res)
            input_ids = []
            attention_mask = []
            labels = []
            instruction = ''
            answer = ''
            for res in res_list:
                input_ids.extend(res["input_ids"])
                attention_mask.extend(res["attention_mask"])
                labels.extend(res["labels"])
                instruction += res["instruction"]
                answer += res["answer"]

            res = dict(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels, instruction=instruction, answer=answer
            )
            res.update(image=image)

            full_res_list.append(res)
        output = self.merge_all_images(full_res_list)
        return output
