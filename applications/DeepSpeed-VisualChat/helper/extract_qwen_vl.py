from transformers import AutoModelForCausalLM
import torch

PATH = "Qwen/Qwen-VL-Chat"

model = AutoModelForCausalLM.from_pretrained(PATH, device_map="cuda", trust_remote_code=True).eval()

state_dict = model.state_dict()
save_dict = {}
for k,v in state_dict.items():
    if 'visual' in k:
        if 'transformer.visual.proj' not in k: # we don't need the proj layer
            save_dict[k.replace('transformer.visual.', '')] = v
torch.save(save_dict, './qwen_clip/pytorch_model.bin')