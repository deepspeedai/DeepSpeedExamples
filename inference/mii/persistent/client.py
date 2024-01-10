import argparse
import mii

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
parser.add_argument(
    "--prompts", type=str, nargs="+", default=["DeepSpeed is", "Seattle is"]
)
parser.add_argument("--max-new-tokens", type=int, default=128)
args = parser.parse_args()

client = mii.client(args.model)
responses = client(
    args.prompts, max_new_tokens=args.max_new_tokens, return_full_text=True
)

for r in responses:
    print(r, "\n", "-" * 80, "\n")
