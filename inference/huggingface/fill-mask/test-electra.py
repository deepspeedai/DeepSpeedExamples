from transformers import pipeline
import transformers
import deepspeed
import torch
import os
from transformers.models.electra.modeling_electra import ElectraLayer

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '4'))

pipe = pipeline('fill-mask', model="google/electra-base-generator",
    tokenizer="google/electra-base-generator")

# The injection_policy shows two things:
#   1. which layer module we need to add Tensor-Parallelism
#   2. the name of one or several linear layers: a) attention_output (both encoder and decoder), 
#       and b) transformer output
pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=torch.float,
    injection_policy={ElectraLayer: ('output.dense')}
)
pipe.device = torch.device(f'cuda:{local_rank}')
output = pipe(f"HuggingFace is creating a {pipe.tokenizer.mask_token} that the community uses to solve NLP tasks.")

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(output)
