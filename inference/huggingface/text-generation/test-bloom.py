import os
import torch
import deepspeed
import transformers

# Pipeline class to mimic HF pipeline
from utils import Pipeline

model_name = 'bigscience/bloom-3b'
dtype = torch.float16
num_tokens = 100

# Get local gpu rank from torch.distributed/deepspeed launcher
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

pipe = Pipeline(model_name=model_name,
                dtype=dtype,
                is_meta=True,
                device=local_rank
)

pipe.model = deepspeed.init_inference(
                pipe.model,
                mp_size=world_size,
                dtype=dtype,
                replace_with_kernel_inject=True,
                base_dir=pipe.repo_root,
                checkpoint=pipe.checkpoints_json
)

output = pipe('DeepSpeed is', num_tokens=num_tokens, do_sample=False)
print(output)
