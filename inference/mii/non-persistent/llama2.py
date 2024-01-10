import mii

pipe = mii.pipeline("meta-llama/Llama-2-7b-hf")
responses = pipe("DeepSpeed is", max_new_tokens=128, return_full_text=True)
if pipe.is_rank_0:
    print(responses[0])
