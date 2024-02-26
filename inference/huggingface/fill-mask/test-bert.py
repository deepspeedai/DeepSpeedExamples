from transformers import pipeline
import transformers
import deepspeed
import torch
import os
import argparse
from deepspeed.accelerator import get_accelerator

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, help="hf model name")
parser.add_argument("--dtype", type=str, default="fp16", help="fp16 or fp32 or bf16")
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
parser.add_argument("--trials", type=int, default=8, help="number of trials")
parser.add_argument("--kernel_inject", action="store_true", help="inject kernels on")
parser.add_argument("--graphs", action="store_true", help="CUDA Graphs on")
parser.add_argument("--triton", action="store_true", help="triton kernels on")
parser.add_argument("--deepspeed", action="store_true", help="use deepspeed inference")
parser.add_argument("--task", type=str, default="fill-mask", help="fill-mask or token-classification")
args = parser.parse_args()

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '4'))

pipe = pipeline('fill-mask', model='bert-large-cased', device=local_rank)

pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=torch.float16 if args.triton else torch.float,
    replace_with_kernel_inject=args.kernel_inject,
    use_triton=args.triton,
)

pipe.device = torch.device(get_accelerator().device_name(local_rank))
output = pipe("In Autumn the [MASK] fall from the trees.")

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(output)
