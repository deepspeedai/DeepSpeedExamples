#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate GSM8K completions using vLLM (multi-GPU tensor parallel).
Outputs JSONL with task_id, question, correct_answer, predicted_answer, and raw_completion.

Usage:
    python evaluate/gsm8k/gen_gsm8k.py --model moonshotai/Moonlight-16B-A3B --output eval_results/gsm8k_baseline --tp 8
"""

import argparse
import json
import os
import re

os.environ.setdefault("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from datasets import load_dataset
from vllm import LLM, SamplingParams


def format_gsm8k_prompt(question):
    return f"### Instruction:\n{question}\n\n### Response:\n"


def extract_answer(text):
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", help="GSM8K split to evaluate")
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n_samples", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "samples.jsonl")

    print(f"Loading model: {args.model} (tp={args.tp})")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_new_tokens,
    )

    print(f"Loading GSM8K dataset (split={args.split})")
    dataset = load_dataset("openai/gsm8k", "main", split=args.split, trust_remote_code=True)
    print(f"Loaded {len(dataset)} examples")

    prompts = []
    meta = []
    for i, example in enumerate(dataset):
        question = example["question"]
        gold_answer = extract_answer(example["answer"])
        prompt = format_gsm8k_prompt(question)
        for _ in range(args.n_samples):
            prompts.append(prompt)
            meta.append({
                "task_id": i,
                "question": question,
                "correct_answer": gold_answer,
            })

    print(f"Generating {len(prompts)} completions")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    with open(out_path, "w") as f_out:
        for m, output in zip(meta, outputs):
            completion = output.outputs[0].text.strip()
            predicted = extract_answer(completion)
            sample = {
                "task_id": m["task_id"],
                "question": m["question"],
                "correct_answer": m["correct_answer"],
                "predicted_answer": predicted,
                "raw_completion": completion,
            }
            f_out.write(json.dumps(sample) + "\n")

    print(f"Done. {len(outputs)} samples in {out_path}")


if __name__ == "__main__":
    main()
