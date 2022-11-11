from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import deepspeed
import math
import os
import torch
from utils import Pipeline


parser = ArgumentParser()

parser.add_argument("--name", required=True, type=str, help="model_name")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--dtype", default="float16", type=str, choices=["float32", "float16", "int8"], help="data-type")
parser.add_argument("--ds_inference", default=True, type=bool, help="enable ds-inference")
parser.add_argument("--max_tokens", default=1024, type=int, help="maximum tokens used for the text-generation KV-cache")
parser.add_argument("--max_new_tokens", default=50, type=int, help="maximum new tokens to generate")
parser.add_argument("--greedy", default=False, type=bool, help="greedy generation mode")
parser.add_argument("--use_meta_tensor", default=False, type=bool, help="use the meta tensors to initialize model")
parser.add_argument("--use_cache", default=True, type=bool, help="use cache for generation")
args = parser.parse_args()

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

data_type = getattr(torch, args.dtype)
pipe = Pipeline(model_name=args.name,
                dtype=data_type,
                is_meta=args.use_meta_tensor,
                device=local_rank
)

if args.ds_inference:
    pipe.model = deepspeed.init_inference(pipe.model,
                                    dtype=data_type,
                                    mp_size=world_size,
                                    replace_with_kernel_inject=True,
                                    max_tokens=args.max_tokens
                                    )

input_sentences = [
         "DeepSpeed is a machine learning framework",
         "He is working on",
         "He has a",
         "He got all",
         "Everyone is happy and I can",
         "The new movie that got Oscar this year",
         "In the far far distance from our galaxy,",
         "Peace is the only way"
]

if args.batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(args.batch_size / len(input_sentences))

inputs = input_sentences[:args.batch_size]

outputs = pipe(inputs, 
              num_tokens=args.max_new_tokens, 
              do_sample=(not args.greedy),
              use_cache=args.use_cache)

for i, o in zip(inputs, outputs):
    print(f"\nin={i}\nout={o}\n{'-'*60}")

