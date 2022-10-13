import os
import torch
import deepspeed
import deepspeed.module_inject as module_inject

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer
from transformers.models.opt.modeling_opt import OPTDecoderLayer

# Get local gpu rank from torch.distributed/deepspeed launcher
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

print(
    "***************** Creating model in RANK ({0}) with WORLD_SIZE = {1} *****************"
    .format(local_rank, world_size))

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=torch.float16)

# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
# model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=torch.float16)

# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
# model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", torch_dtype=torch.float16)

# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
# model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b", torch_dtype=torch.float16)

# tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-13b')
# model = AutoModelForCausalLM.from_pretrained("facebook/opt-13b", torch_dtype=torch.float16)

generator = pipeline('text-generation',
                     model=model,
                     tokenizer=tokenizer,
                     device=local_rank)

generator.model = deepspeed.init_inference(generator.model,
                                           mp_size=world_size,
                                           dtype=torch.float16,
                                           replace_with_kernel_inject=True)

string = generator("DeepSpeed is", do_sample=True, use_cache=True)
print(string)
