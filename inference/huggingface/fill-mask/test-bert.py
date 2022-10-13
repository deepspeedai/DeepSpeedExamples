from transformers import pipeline
import transformers
import deepspeed
import torch
import os
from transformers.models.bert.modeling_bert import BertLayer

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '4'))

pipe = pipeline('fill-mask', model='bert-large-cased', device=local_rank)

pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=torch.float,
    injection_policy={BertLayer : ('output.dense')}
)

pipe.device = torch.device(f'cuda:{local_rank}')
output = pipe("In Autumn the [MASK] fall from the trees.")

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(output)
