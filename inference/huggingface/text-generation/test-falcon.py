from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
import torch
import deepspeed
import time
from deepspeed.accelerator import get_accelerator

model = "tiiuae/falcon-40b"

tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)


model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True).bfloat16()
model = deepspeed.init_inference(model, mp_size=4)


input_prompt = [
   "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:"
   ]

input_tokens = tokenizer.batch_encode_plus(input_prompt, return_tensors="pt",)

for t in input_tokens:
    if torch.is_tensor(input_tokens[t]):
        input_tokens[t] = input_tokens[t].to(get_accelerator().current_device_name())
input_tokens.pop('token_type_ids')

sequences = model.generate(**input_tokens, min_length=200, max_length=300, do_sample=True)

print(f"Result: {tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]}")
