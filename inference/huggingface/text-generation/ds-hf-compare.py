import torch
import deepspeed
from transformers import pipeline
from difflib import SequenceMatcher
from argparse import ArgumentParser
from deepspeed.accelerator import get_accelerator

parser = ArgumentParser()

parser.add_argument("--model", required=True, type=str, help="model_name")
parser.add_argument("--dtype", default="float16", type=str, choices=["float32", "float16", "int8", "bfloat16"], help="data-type")
parser.add_argument("--num_inputs", default=1, type=int, help="number of test inputs")
parser.add_argument("--min_length", default=200, type=int, help="minimum tokens generated")
parser.add_argument("--max_length", default=300, type=int, help="maximum tokens generated")
parser.add_argument("--print_outputs", action='store_true', help="print generated text outputs")
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
parser.add_argument("--use_kernel", action='store_true', help="enable kernel-injection")
args = parser.parse_args()

def print_0(output):
    if args.local_rank == 0:
        print(output)

def string_similarity(str1, str2):
    matcher = SequenceMatcher(None, str1, str2)
    similarity_ratio = matcher.ratio()
    return similarity_ratio

test_inputs = [
    "This is a test",
    "One fish, two fish, red fish,",
    "Microsoft is in Washington DC",
    "The ancient art of basket weaving",
    "Large language models are useful",
    "You shouldn't buy a car without first checking",
    "In today's lesson, we will cover the conflict between",
    "Interestingly, the humble bumblebee is essential to our",
    "How many blue buttons and yellow marbles are left in the",
    "My favorite band is playing at a local music festival next month",
    "Fortunately, I made it just in time to the event to tell her that",
    "Once upon a time in a galaxy far away, there lived a boy named Anakin who",
    "It is projected that by the year 3035, there will be more humans on the planet than ",
    "Many years ago, we were hiking in the Amazon rain forest when we stumbled upon an impressive",
    "Let's discuss today's agenda. First, we will go around and introduce ourselves. Next, we will cover our 3 essential markers for success: 1) ",
    "These two historical figures ",
    "I saw a news article about a major scientific discovery ",
    "A poem about the beauty of the night sky",
    "Improving mental health ",
    "Being a professional athlete",
    "There are many exotic travel destinations",
    "She needed a recipe for a unique and delicious dessert",
    "The process of creating a work of art",
    "The importance of renewable energy has been a popular topic among",
    "Hiking to the top of a mountain is no easy task. It can takes several hours and ",
    "His latest clothing collection was all the rave at the last year's fashion week. Several ",
    "Here's a list of 10 thought-provoking discussion questions",
    "The show last night had to be postponed due to weather. I heard that people waited hours in the rain",
    "A successful small business can be evaluated these three performance metrics",
    "My favorite motivational quotes to inspire others are",
    "A magical creature living in a hidden forest",
    "The preparation of a gourmet meal",
    "I overheard two scientists discussing a groundbreaking scientific theory",
    "He wrote a blog post about the benefits of mindfulness and meditation.",
    "This set of instructions for assembling a piece of furniture",
    "Training for a marathon",
    "What are your hopes and dreams for the world?",
    "Imagine you are a time traveler. Write a journal entry about your visit to a historical event.",
    "Generate a list of 10 unique and exciting travel destinations.",
    "She gave speech advocating for equal rights",
    "The process of producing a documentary film ",
    "With a flick of a wand, the magician made the rabbit disappear",
    "The bustling marketplace was a kaleidoscope of colors and sounds. There were at least 100 vendors and dozens of"
]

if args.num_inputs < len(test_inputs):
    inputs = test_inputs[:args.num_inputs]
else:
    print_0(f"Warning: num_inputs ({args.num_inputs}) is greater than the number of test inputs ({len(test_inputs)}). Using all test inputs.")
    inputs = test_inputs

data_type = getattr(torch, args.dtype)
pipe = pipeline('text-generation', args.model, torch_dtype=data_type, device=torch.device(get_accelerator().device_name(0)))

base_out_list = []
match_count=0
mismatch_count=0

# Run the baseline model
if args.local_rank == 0:
    for prompt in inputs:
        base_out_list += pipe(prompt, do_sample=False, min_length=args.min_length, max_length=args.max_length)

# Initialize the model with DeepSpeed
pipe.model = deepspeed.init_inference(pipe.model, dtype=data_type, replace_with_kernel_inject=args.use_kernel)

# Run the DeepSpeed model and compare outputs
for prompt, base_out in zip(inputs, base_out_list):
    ds_out = pipe(prompt, do_sample=False, min_length=args.min_length, max_length=args.max_length)
    if args.local_rank == 0:
        if args.print_outputs:
            print(f"baseline output: {base_out}")
            print(f"deepspeed output: {ds_out}")
            print(f"{'-'*60}")
        if base_out == ds_out[0]:
            if args.print_outputs: print("outputs match")
            match_count += 1
        else:
            if args.print_outputs: print("outputs do not match")
            mismatch_count += 1
        similarity = string_similarity(base_out['generated_text'], ds_out[0]['generated_text'])
        if args.print_outputs: print(f"The similarity ratio is: {similarity*100}%")

print_0(f"Matches: {match_count}\nMismatches: {mismatch_count}")
