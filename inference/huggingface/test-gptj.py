import os
import torch
import deepspeed
import transformers

from deepspeed import module_inject
from transformers import pipeline
from transformers.models.gptj.modeling_gptj import GPTJBlock

# Get local gpu rank from torch.distributed/deepspeed launcher
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

print(
    "***************** Creating model in RANK ({0}) with WORLD_SIZE = {1} *****************"
    .format(local_rank,
            world_size))
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

inp_tokens = tokenizer("DeepSpeed is", return_tensors="pt",)
model = deepspeed.init_inference(model,
                                 mp_size=world_size,
                                 dtype=torch.half,
                                 replace_with_kernel_inject=True)
                                 
for token in inp_tokens:
    if torch.is_tensor(inp_tokens[token]):
        inp_tokens[token] = inp_tokens[token].to(f'cuda:{local_rank}')
        
model.cuda().to(f'cuda:{local_rank}')
string = tokenizer.batch_decode(model.generate(**inp_tokens,min_length=50,max_length=50,do_sample=True))[0]
print(string)
