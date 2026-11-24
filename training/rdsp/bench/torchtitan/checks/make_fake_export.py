# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""A small export in the format of bench/mimo/export_bridge_batches.py, built with the
real Qwen3.5 Hugging Face processor: random images of varying sizes, a chat prompt and
an answer, labels on the last 12 real tokens (Hugging Face convention: unshifted, -100
elsewhere; any known mask serves the check), every row padded to 2048. Row i's answer
names i, so order can be checked.

    .venv/bin/python rdsp/bench/torchtitan/checks/make_fake_export.py <out dir> [steps]
"""

import os
import sys

import huggingface_hub
import numpy as np
import torch
import transformers
from PIL import Image

SEQ, ROWS = 2048, 64
out = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
path = huggingface_hub.snapshot_download(
    "Qwen/Qwen3.5-27B", allow_patterns=["*.json", "*.txt", "*.jinja"]
)
processor = transformers.AutoProcessor.from_pretrained(path)
pad = processor.tokenizer.convert_tokens_to_ids("<|endoftext|>")
rng = np.random.default_rng(0)
os.makedirs(out, exist_ok=True)
for step in range(steps):
    samples = []
    for r in range(ROWS):
        i = step * ROWS + r
        h, w = (int(x) for x in rng.integers(48, 200, size=2))
        image = Image.fromarray(rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8))
        prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Read the receipt."},
                ],
            }
        ]
        full = prompt + [
            {"role": "assistant", "content": [{"type": "text", "text": f"row {i}: total {i * 7}"}]}
        ]
        f = processor.apply_chat_template(
            full, tokenize=True, return_dict=True, return_tensors="pt"
        )
        ids = f["input_ids"][0]
        n = ids.shape[0]
        start = n - 12
        input_ids = torch.full((1, SEQ), pad, dtype=ids.dtype)
        input_ids[0, :n] = ids
        attention_mask = torch.zeros((1, SEQ), dtype=torch.long)
        attention_mask[0, :n] = 1
        labels = torch.full((1, SEQ), -100, dtype=ids.dtype)
        labels[0, start:n] = ids[start:n]
        mm = f.get("mm_token_type_ids")
        mm_ids = torch.zeros((1, SEQ), dtype=torch.long)
        if mm is not None:
            mm_ids[0, :n] = mm[0]
        samples.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "mm_token_type_ids": mm_ids,
                "pixel_values": f["pixel_values"].to(torch.bfloat16),
                "image_grid_thw": f["image_grid_thw"].clone(),
                "labels": labels,
                "tokens": int((labels != -100).sum()),
            }
        )
    torch.save(samples, os.path.join(out, f"step_{step:05d}.pt"))
print(f"wrote {steps} steps x {ROWS} rows to {out}; processor {path}")
