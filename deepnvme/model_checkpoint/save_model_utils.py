import argparse
import os
from transformers import AutoModelForCausalLM
from transformers import T5ForConditionalGeneration
from torch_save_utils import PINNED_BUFFER_MB


GPT2L = 'gpt2-large'
TINY_T5 = 'tiny-t5'
PHI3_MINI = 'phi3'
PHI3_VISION = 'phi3-v'
LLAMA3_1B = 'llama3-1B'

HF_MODELS_DICT = {
    TINY_T5: "hf-internal-testing/tiny-random-t5",
    GPT2L: GPT2L,
    PHI3_MINI: "microsoft/Phi-3.5-mini-instruct",
    PHI3_VISION: "microsoft/Phi-3.5-vision-instruct",
    LLAMA3_1B: "meta-llama/Llama-3.2-1B",
}

def _get_hf_model(tag):
    model_name = HF_MODELS_DICT[tag]
    if tag == TINY_T5:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    return model, model_name, tag

def get_model(model_tag):
    return _get_hf_model(model_tag)


def validate_arguments(args):
    success = True

    if not args.model in HF_MODELS_DICT:
        print(f'{args.model} is not a supported HF model tag')
        success = False

    if args.optimizer and args.half:
        if not args.gpu:
            print(f'mixed precision only supported with gpu tensors')
            success = False

    return success


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',
                        default=None,
                        type=str,
                        required=True,
                        help='Folder to use for I/O.')

    parser.add_argument(
        '--model',
        default=None,
        type=str,
        required=True,
        help=f'HuggingFace tag of model. Available models = {list(HF_MODELS_DICT.keys())}')

    parser.add_argument('--local_rank',
                        type=int,
                        default=0,
                        help='Local rank' )

    parser.add_argument('--legacy',
                        action='store_true',
                        help='Use torch legacy save format')

    parser.add_argument('--optimizer',
                        action='store_true',
                        help='Include optimizer state in checkpoint.')

    parser.add_argument('--fused',
                        action='store_true',
                        help='Use fused fp16 optimizer.')

    parser.add_argument('--gpu', action='store_true', help='Use gpu tensors.')

    parser.add_argument('--half',
                        action='store_true',
                        help='Use half-precision tensors.')

    parser.add_argument(
        '--io_buffer_mb',
        type=int,
        default=PINNED_BUFFER_MB,
        help=f'Size of pinned i/o buffer in MB. Default = {PINNED_BUFFER_MB}')

    parser.add_argument('--zero_stage',
                        type=int,
                        default=0,
                        help='ZeRO optimization stage. Default = 0')

    parser.add_argument('--cpu_offload',
                        action='store_true',
                        help='Enable CPU offload of optimizer state.')

    parser.add_argument('--no-statistics',
                        action='store_true',
                        help='Suppress low-level performance statistics.')

    parser.add_argument('--single_io_buffer',
                        action='store_true',
                        help='Disable double buffering of i/o buffer.')


    #parser.add_argument('--single_writer', action='store_true', help='Disable parallel rank writes of data parallel (replicated) state')

    args = parser.parse_args()
    print(f'args = {args}')
    return args
