import deepspeed
import torch
import os
from local_pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline as StableDiffusionPipelineBaseline
import argparse


seed = 123450011
parser = argparse.ArgumentParser()
parser.add_argument("--finetuned_model", default="./sd-distill-lora-multi-50k-50", type=str, help="Path to the fine-tuned model")
parser.add_argument("--base_model", default="stabilityai/stable-diffusion-2-1-base", type=str, help="Path to the baseline model")
parser.add_argument("--out_dir", default="image_out/", type=str, help="Path to the generated images")
parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance Scale')
parser.add_argument("--use_local_pipe", action='store_true', help="Use local SD pipeline")
parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank")
args = parser.parse_args()


local_rank = int(os.getenv("LOCAL_RANK", "0"))
device = torch.device(f"cuda:{local_rank}")
world_size = int(os.getenv('WORLD_SIZE', '1'))


if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"Directory '{args.out_dir}' has been created to store the generated images.")
else:
        print(f"Directory '{args.out_dir}' already exists and stores the generated images.")


prompts = ["A boy is watching TV",
           "A photo of a person dancing in the rain",
           "A photo of a boy jumping over a fence",
           "A photo of a boy is kicking a ball",
           "A beach with a lot of waves on it",
           "A road that is going down a hill",
           "3d rendering of 5 tennis balls on top of a cake",
           "A person holding a drink of soda",
           "A person is squeezing a lemon",
           "A person holding a cat"]


# Load the pipelines
pipe_new = StableDiffusionPipeline.from_pretrained(args.base_model, torch_dtype=torch.float16).to("cuda")
pipe_baseline = StableDiffusionPipelineBaseline.from_pretrained(args.base_model, torch_dtype=torch.float16).to("cuda")

pipe_new.scheduler = DPMSolverMultistepScheduler.from_config(pipe_new.scheduler.config)
pipe_baseline.scheduler = DPMSolverMultistepScheduler.from_config(pipe_baseline.scheduler.config)

# Load the Lora weights
pipe_new.unet.load_attn_procs(args.finetuned_model)

pipe_new = deepspeed.init_inference(pipe_new, mp_size=world_size, dtype=torch.half)
pipe_baseline = deepspeed.init_inference(pipe_baseline, mp_size=world_size, dtype=torch.half)

# Generate the images
for prompt in prompts:
        #--- baseline image
        generator = torch.Generator("cuda").manual_seed(seed)
        image_baseline = pipe_baseline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        image_baseline.save(args.out_dir+"BASELINE_seed_"+str(seed)+"_"+prompt[0:100]+".png")

        #--- new image
        generator = torch.Generator("cuda").manual_seed(seed)
        image_new = pipe_new(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        image_new.save(args.out_dir+"NEW_seed_"+str(seed)+"_"+prompt[0:100]+".png")
