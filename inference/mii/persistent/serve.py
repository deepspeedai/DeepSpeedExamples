import argparse
import mii

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
parser.add_argument("--tensor-parallel", type=int, default=1)
args = parser.parse_args()

mii.serve(args.model, tensor_parallel=args.tensor_parallel)

print(f"Serving model {args.model} on {args.tensor_parallel} GPU(s).")
print(f"Run `python client.py --model {args.model}` to connect.")
print(f"Run `python terminate.py --model {args.model}` to terminate.")
