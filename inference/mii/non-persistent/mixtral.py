import mii

pipe = mii.pipeline("mistralai/Mixtral-8x7B-v0.1")
responses = pipe("DeepSpeed is", max_new_tokens=128, return_full_text=True)
if pipe.is_rank_0:
    print(responses[0])
