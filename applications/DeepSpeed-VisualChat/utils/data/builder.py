# This file is adapted from https://github.com/open-mmlab/Multimodal-GPT

import numpy as np
import torch

from .aokvqa_dataset import AOKVQADataset  # noqa: F401
from .cc_sbu_align_dataset import CcSbuAlignDataset  # noqa: F401
from .coco_caption_dataset import COCOCaptionDataset  # noqa: F401
from .dial_dataset import DialDataset  # noqa: F401
from .llava_dataset import LlavaDataset  # noqa: F401
from .llava_otter_blend_dataset import LlavaOtterBlendDataset  # noqa: F401
from .ocr_vqa_dataset import OCRVQADataset  # noqa: F401
from .otter_mimicit_cgd_dataset import OtterMimicitCgdDataset  # noqa: F401
from .otter_mimicit_sd_dataset import OtterMimicitSdDataset  # noqa: F401
from .otter_mimicit_sn_dataset import OtterMimicitSnDataset  # noqa: F401
from .otter_mimicit_tvc_dataset import OtterMimicitTvcDataset  # noqa: F401
from .otter_mimicit_vst_dataset import OtterMimicitVstDataset  # noqa: F401
from .sparkles_dialogue_dataset import SparklesDialogueDataset  # noqa: F401
from .vqa_dataset import ConcatDataset  # noqa: F401
from utils.utils import print_rank_0


def build_dataset(data_path, data_debug_path, dataset_name, dataset_sample,
                  dataset_concatenate_samples, max_num_image_per_sample, **kwargs):
    if isinstance(dataset_name, list):
        datasets = [build_dataset(data_path, data_debug_path,
                                  dataset_name[i], dataset_sample[i],
                                  dataset_concatenate_samples[i],
                                  max_num_image_per_sample,
                                  **kwargs) for i in range(len(dataset_name))]
        return ConcatDataset(datasets)
    if dataset_name == "aokvqa":
        dataset = AOKVQADataset(
            data_path,
            data_debug_path,
            dataset_concatenate_samples,
            **kwargs,
        )
    elif dataset_name == "coco_caption":
        dataset = COCOCaptionDataset(
            data_path,
            data_debug_path,
            dataset_concatenate_samples,
            **kwargs,
        )
    elif dataset_name == "llava":
        dataset = LlavaDataset(
            data_path,
            data_debug_path,
            dataset_concatenate_samples,
            **kwargs,
        )
    elif dataset_name == "llava_dial":
        dataset = DialDataset(
            dataset_name,
            data_path,
            data_debug_path,
            dataset_concatenate_samples,
            **kwargs,
        )
    elif dataset_name == "llava_otter_blend":
        dataset = LlavaOtterBlendDataset(
            data_path,
            data_debug_path,
            dataset_concatenate_samples,
            followup=False,
            **kwargs,
        )
    elif dataset_name == "minigpt4":
        dataset = CcSbuAlignDataset(
            data_path,
            data_debug_path,
            dataset_concatenate_samples,
            **kwargs,
        )
    elif dataset_name == "ocr_vqa":
        dataset = OCRVQADataset(
            data_path,
            data_debug_path,
            dataset_concatenate_samples,
            **kwargs,
        )
    elif dataset_name == "otter_mimicit_cgd":
        dataset = OtterMimicitCgdDataset(
            data_path,
            data_debug_path,
            dataset_concatenate_samples,
            **kwargs,
        )
    elif dataset_name == "otter_mimicit_sd":
        dataset = OtterMimicitSdDataset(
            data_path,
            data_debug_path,
            dataset_concatenate_samples,
            **kwargs,
        )
    elif dataset_name == "otter_mimicit_sn":
        dataset = OtterMimicitSnDataset(
            data_path,
            data_debug_path,
            dataset_concatenate_samples,
            max_num_image_per_sample,
            **kwargs,
        )
    elif dataset_name == "otter_mimicit_tvc":
        dataset = OtterMimicitTvcDataset(
            data_path,
            data_debug_path,
            dataset_concatenate_samples,
            max_num_image_per_sample,
            **kwargs,
        )
    elif dataset_name == "otter_mimicit_vst":
        dataset = OtterMimicitVstDataset(
            data_path,
            data_debug_path,
            dataset_concatenate_samples,
            max_num_image_per_sample,
            **kwargs,
        )
    elif dataset_name == "sparkles_dialogue":
        dataset = SparklesDialogueDataset(
            data_path,
            data_debug_path,
            dataset_concatenate_samples,
            **kwargs,
        )
    else:
        raise NotImplementedError

    if dataset_sample != 'all':
        dataset_sample = int(dataset_sample)
        random_indices = np.random.choice(len(dataset), min(dataset_sample, len(dataset)), replace=False)
        subsample_dataset = torch.utils.data.Subset(dataset, random_indices)
        subsample_dataset.collater = dataset.collater
        print_rank_0(f"[DATA] Built dataset {dataset_name} with {len(subsample_dataset)} samples.")
        return subsample_dataset
    else:
        print_rank_0(f"[DATA] Built dataset {dataset_name} with all {len(dataset)} samples.")
        return dataset
