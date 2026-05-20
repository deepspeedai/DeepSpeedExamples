#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate MMLU completions using vLLM (multi-GPU tensor parallel).
Outputs JSONL with task_id, question, choices, correct_answer, and predicted_answer.

Usage:
    python evaluate/mmlu/gen_mmlu.py --model moonshotai/Moonlight-16B-A3B --output eval_results/mmlu_baseline --tp 8
"""

import argparse
import json
import os

os.environ.setdefault("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from datasets import load_dataset
from vllm import LLM, SamplingParams

LABELS = "ABCDEFGHIJ"


def format_mmlu_prompt(question, choices):
    prompt = f"### Instruction:\n{question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{LABELS[i]}. {choice}\n"
    prompt += "\nAnswer with the letter of the correct choice.\n\n### Response:\n"
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--subset", type=str, default="all", help="MMLU subset name")
    parser.add_argument("--split", type=str, default="test", help="MMLU split to evaluate")
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size")
    parser.add_argument("--max_new_tokens", type=int, default=10)
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

    print(f"Loading MMLU dataset (subset={args.subset}, split={args.split})")
    dataset = load_dataset("cais/mmlu", args.subset, split=args.split, trust_remote_code=True)
    print(f"Loaded {len(dataset)} examples")

    prompts = []
    meta = []
    for i, example in enumerate(dataset):
        question = example["question"]
        choices = example["choices"]
        answer = example["answer"]
        if isinstance(answer, int):
            answer = LABELS[answer]
        prompt = format_mmlu_prompt(question, choices)
        for _ in range(args.n_samples):
            prompts.append(prompt)
            meta.append({
                "task_id": i,
                "question": question,
                "choices": choices,
                "correct_answer": answer,
            })

    print(f"Generating {len(prompts)} completions")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    with open(out_path, "w") as f_out:
        for m, output in zip(meta, outputs):
            completion = output.outputs[0].text.strip()
            predicted = ""
            for ch in completion:
                if ch.upper() in LABELS[: len(m["choices"])]:
                    predicted = ch.upper()
                    break
            sample = {
                "task_id": m["task_id"],
                "question": m["question"],
                "choices": m["choices"],
                "correct_answer": m["correct_answer"],
                "predicted_answer": predicted,
                "raw_completion": completion,
            }
            f_out.write(json.dumps(sample) + "\n")

    print(f"Done. {len(outputs)} samples in {out_path}")


if __name__ == "__main__":
    main()
