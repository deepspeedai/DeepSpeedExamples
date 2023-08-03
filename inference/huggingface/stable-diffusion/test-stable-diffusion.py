import deepspeed
import torch
import os
from local_pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers import DiffusionPipeline
import argparse

# In this example the SD inference pipeline is optimized based on recommendations in the research paper
# titled "Selective Guidance: Are All the Denoising Steps of Guided Diffusion Important?"(https://arxiv.org/abs/2305.09847).
# In this method, the last iterations of the denoising loop are optimized by simplifying the noise computation. In the guided diffusion
# process of the SD pipeline, the computed noise consists of two components: conditional noise and unconditional noise.
# However, including both components of noise doubles the computational requirements for running the denoising loop.
# To address this issue, the optimized iterations in this approach eliminate the unconditional noise, thereby reducing the
# compute cost associated with the generation process.
# The user has the flexibility to choose the percentage of iterations they want to optimize (opt_percentage). As well as
# setting the prompt and the guidance_scale.

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="prompthero/midjourney-v4-diffusion", type=str, help="model_name")
parser.add_argument('--prompt', type=str, default='A dog on a rocket', help='The prompt to use in creating the image')
parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance Scale')
parser.add_argument('--opt_percentage', type=int, default=0, help='The percentage of the last iterations to be optimized')
parser.add_argument("--use_local_pipe", action='store_true', help="Use local SD pipeline")
parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank")
args = parser.parse_args()

model = args.name
local_rank = int(os.getenv("LOCAL_RANK", "0"))
device = torch.device(f"cuda:{local_rank}")
world_size = int(os.getenv('WORLD_SIZE', '1'))
generator = torch.Generator(device=torch.cuda.current_device())

if args.use_local_pipe:
    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.half)
else:
    pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.half)
pipe = pipe.to(device)

seed = 0xABEDABE7
generator.manual_seed(seed)
if args.use_local_pipe:
    baseline_image = pipe(args.prompt, guidance_scale=args.guidance_scale, generator=generator, opt_percentage=0).images[0]
else:
    baseline_image = pipe(args.prompt, guidance_scale=args.guidance_scale, generator=generator).images[0]
baseline_image.save(f"baseline.png")


# NOTE: DeepSpeed inference supports local CUDA graphs for replaced SD modules.
#       Local CUDA graphs for replaced SD modules will only be enabled when `mp_size==1`
pipe = deepspeed.init_inference(
    pipe,
    mp_size=world_size,
    dtype=torch.half,
    replace_with_kernel_inject=True,
    enable_cuda_graph=True if world_size==1 and not args.use_local_pipe else False,
    )

generator.manual_seed(seed)
if args.use_local_pipe:
    deepspeed_image = pipe(args.prompt, guidance_scale=args.guidance_scale, generator=generator, opt_percentage=args.opt_percentage).images[0]
else:
    deepspeed_image = pipe(args.prompt, guidance_scale=args.guidance_scale, generator=generator).images[0]
deepspeed_image.save(f"deepspeed.png")
