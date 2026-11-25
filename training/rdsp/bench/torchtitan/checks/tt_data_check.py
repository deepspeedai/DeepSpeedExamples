# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""CPU check of cord_data.py through TorchTitan's own loader and MultiModalCollator:
every microbatch against the exported row it came from. Saves the first rows' M-RoPE
positions for checks/compare_mrope.py (Hugging Face's get_rope_index).

    PYTHONPATH=rdsp/bench/torchtitan <torchtitan venv>/bin/python \\
        rdsp/bench/torchtitan/checks/tt_data_check.py <export dir> <HF Qwen3.5 snapshot> <out.pt>
"""

import glob
import os
import sys

import cord_data
import torch
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import MultiModalTokenizer

export, hf, out = sys.argv[1:4]
# torchtitan/models/qwen3_5/flavors.py QWEN3_5_SPECIAL_TOKENS (importing the model
# package pulls GPU-only kernels)
SPECIAL = {
    "image_token": "<|image_pad|>",
    "video_token": "<|video_pad|>",
    "vision_start_token": "<|vision_start|>",
    "vision_end_token": "<|vision_end|>",
    "pad_token": "<|endoftext|>",
}
tokenizer = MultiModalTokenizer.Config(**SPECIAL).build(tokenizer_path=hf)
loader = cord_data.exported_dataloader(export).build(
    dp_world_size=1,
    dp_rank=0,
    tokenizer=tokenizer,
    max_context_length=2048,
    num_tokens_per_microbatch=2048,
)
rows = [
    r
    for p in sorted(glob.glob(os.path.join(export, "step_*.pt")))
    for r in torch.load(p, weights_only=False)
]
saved, n = [], 0
for mb, row in zip(loader, rows):
    ids, hf_labels = row["input_ids"][0], row["labels"][0]
    assert torch.equal(mb.input.flatten(), ids), f"row {n}: input differs"
    want = torch.full_like(ids, IGNORE_INDEX)
    want[:-1] = hf_labels[1:]
    assert torch.equal(mb.labels.flatten(), want), f"row {n}: labels differ"
    kw = mb.model_kwargs
    assert torch.equal(kw["pixel_values"], row["pixel_values"]), f"row {n}: patches differ"
    assert torch.equal(kw["grid_thw"], row["image_grid_thw"]), f"row {n}: grid differs"
    assert int(mb.loss_token_counts) == int((hf_labels[1:] != IGNORE_INDEX).sum()), (
        f"row {n}: count"
    )
    assert not bool(mb.padding_mask.any()), f"row {n}: unexpected padding"
    assert kw["special_tokens"]["image_id"] == tokenizer.image_id
    if n < 8:
        saved.append(
            {
                "input_ids": ids,
                "image_grid_thw": row["image_grid_thw"],
                "attention_mask": row["attention_mask"][0],
                "tt_mrope": kw["mrope_positions"],
            }
        )
    n += 1
assert n == len(rows), f"loader gave {n} microbatches for {len(rows)} rows"
torch.save(saved, out)
m = saved[0]["tt_mrope"]
print(
    f"OK: {n} microbatches = {len(rows)} exported rows, in order; inputs, shifted labels, "
    f"patches, grids, loss-token counts all equal. mrope_positions shape {tuple(m.shape)}, "
    f"image_id {tokenizer.image_id}, pad_id {tokenizer.pad_id}"
)
