from transformers import pipeline
import transformers
import deepspeed
import torch
import os
from transformers.models.roberta.modeling_roberta import RobertaLayer
from deepspeed.accelerator import get_accelerator

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '4'))

pipe = pipeline('fill-mask', model="roberta-large", device=local_rank)

# The injection_policy shows two things:
#   1. which layer module we need to add Tensor-Parallelism
#   2. the name of several linear layers: a) attention_output (both encoder and decoder), 
#       and b) transformer output

pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=torch.float,
    injection_policy={RobertaLayer: ('output.dense')}
)

pipe.device = torch.device(get_accelerator().device_name(local_rank))
output = pipe("The invention of the <mask> revolutionized the way we communicate with each other.")

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(output)
