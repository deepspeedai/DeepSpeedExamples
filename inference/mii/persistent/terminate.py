import argparse
import mii

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
args = parser.parse_args()

client = mii.client(args.model)
client.terminate_server()

print(f"Terminated server for model {args.model}.")
