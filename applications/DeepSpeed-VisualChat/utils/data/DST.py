from typing import Iterable
import random
import numpy as np
## the following codes are adopted from https://github.com/haotian-liu/LLaVA
## the following codes are adopted from https://github.com/open-mmlab/Multimodal-GPT 
## the following codes are adopted from https://github.com/Luodian/Otter/

# deepspeed template

DEFAULT_SYSTEM_TOKEN="### System instuction:"
DEFAULT_PROMPT = f"You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\n"

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_HUMAN_TOKEN = "### Human:"
DEFAULT_HUMAN_QUESTION_PRETOKEN = "### Question:"
DEFAULT_QUESTION_TOKEN = "<question>"
DEFAULT_HUMAN_IMAGE_PRETOKEN = "### Image:"

DEFAULT_ASSISTANT_TOKEN = "### Answer:"
DEFAULT_ANSWER_TOKEN = "<answer>"

DEFAULT_ASSISTANT_END_ROUND_TOKEN="<endofchunk>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

IMAGE_NUM = '<image#x>'
IMAGE_NUM_1 = '### Image 1:'
IMAGE_NUM_2 = '### Image 2:'
IMAGE_NUM_3 = '### Image 3:'
IMAGE_NUM_4 = '### Image 4:'
IMAGE_NUM_5 = '### Image 5:'
IMAGE_NUM_6 = '### Image 6:'
IMAGE_NUM_7 = '### Image 7:'
IMAGE_NUM_8 = '### Image 8:'

# fow now we at most support 8 images, can be extended to more
image_mapping_dict = {"default": DEFAULT_HUMAN_IMAGE_PRETOKEN, "1": IMAGE_NUM_1, "2": IMAGE_NUM_2, "3": IMAGE_NUM_3, "4": IMAGE_NUM_4, "5": IMAGE_NUM_5, "6": IMAGE_NUM_6, "7": IMAGE_NUM_7, "8": IMAGE_NUM_8}

special_token_list = [DEFAULT_HUMAN_IMAGE_PRETOKEN, DEFAULT_IMAGE_TOKEN] # used for easy image # replacement

DEFAULT_LABEL_PADDING_NUM = -100

def add_special_token(tokenizer):
    tokenizer.add_tokens(special_token_list, special_tokens=True)
    if tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    return tokenizer

def get_image_num_map(tokenizer):
    image_num_map = {}
    for key in image_mapping_dict:
        image_num_map[image_mapping_dict[key]] = tokenizer(image_mapping_dict[key])['input_ids'][1:] # remove <s>
    image_num_map[DEFAULT_HUMAN_IMAGE_PRETOKEN] = image_num_map[DEFAULT_HUMAN_IMAGE_PRETOKEN][0] # convert list to number 
    return image_num_map

TEMPLATE = {
    "description": "Template Modified by DeepSpeed Team for Chat.",
    "prompt_qa_with_image": f'''{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n\n{DEFAULT_HUMAN_QUESTION_PRETOKEN}\n{DEFAULT_QUESTION_TOKEN}\n\n{DEFAULT_ASSISTANT_TOKEN}\n''',
    "prompt_qa_without_image": f'''{DEFAULT_HUMAN_QUESTION_PRETOKEN}\n{DEFAULT_QUESTION_TOKEN}\n\n{DEFAULT_ASSISTANT_TOKEN}\n''',
}

class Prompter:
    def __call__(self, question, with_image=True, first_message=False, num_images=-1, options=None):
        if options:
            raise NotImplementedError("options not supported yet")
            options = ", ".join(options)
            res = TEMPLATE["prompt_choice"].format(image=DEFAULT_IMAGE_TOKEN, question=question, options=options)
        else:
            if with_image:
                res = TEMPLATE["prompt_qa_with_image"].replace(DEFAULT_QUESTION_TOKEN, question)
                if num_images >= 1:
                    tmp_dict = {
                        1: f"{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n\n",
                        2: f"{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n\n",
                        3: f"{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n\n",
                        4: f"{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n\n",
                        5: f"{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n\n",
                        6: f"{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n\n",
                        7: f"{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n\n",
                        8: f"{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n\n",
                    }
                    res = res.replace(f"{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n\n", tmp_dict[num_images])
            else:
                res = TEMPLATE["prompt_qa_without_image"].replace(DEFAULT_QUESTION_TOKEN, question)
            
            if first_message:
                res = DEFAULT_PROMPT + res
        return res

    def get_response(self, output: str) -> str:
        return output.split(TEMPLATE["response_split"])[-1].strip()

def _flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

def flatten(items):
    return list(_flatten(items))


def split_list_with_random_num_items_up_to_a_certain_number(input_list, max_num):
    if len(input_list) <= max_num:
        return [input_list]
    else:
        random_num = random.randint(1, max_num)
        return [input_list[:random_num]] + split_list_with_random_num_items_up_to_a_certain_number(input_list[random_num:], max_num)
            
def random_grouping(input_list, max_num):
    random.shuffle(input_list)
    random_num = np.random.randint(1, max_num+1, len(input_list))
    # use bisect to find the index of random_num, whose sum is equal or large to len(input_list)
    # then split the input_list into groups
    cum_sum = np.cumsum(random_num)
    # find the index now
    left = 0
    right = len(cum_sum) - 1
    while left < right:
        mid = (left + right) // 2
        if cum_sum[mid] >= len(input_list):
            right = mid
        else:
            left = mid + 1
    index = left
    cum_sum = list(cum_sum[:index+1])
    if cum_sum[-1] > len(input_list):
        cum_sum[-1] = len(input_list)
    elif cum_sum[-1] < len(input_list):
        cum_sum.append(len(input_list))

    return [input_list[cum_sum[i]:cum_sum[i+1]] for i in range(len(cum_sum)-1)]
    # return split_list_with_random_num_items_up_to_a_certain_number(input_list, max_num)