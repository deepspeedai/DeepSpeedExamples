# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""TorchTitan's M-RoPE positions (saved by checks/tt_data_check.py) against Hugging
Face's Qwen3_5Model.get_rope_index on the same exported rows, real tokens only
(padding positions never feed a real token: attention is causal).

    .venv/bin/python rdsp/bench/torchtitan/checks/compare_mrope.py <tt_mrope.pt> <export dir>
"""

import glob
import os
import sys

import huggingface_hub
import torch
import transformers

saved = torch.load(sys.argv[1], weights_only=False)
rows = torch.load(sorted(glob.glob(os.path.join(sys.argv[2], "step_*.pt")))[0], weights_only=False)
path = huggingface_hub.snapshot_download("Qwen/Qwen3.5-27B", allow_patterns=["*.json"])
config = transformers.AutoConfig.from_pretrained(path)
with torch.device("meta"):
    model = transformers.Qwen3_5Model(config)
for i, (s, row) in enumerate(zip(saved, rows)):
    assert torch.equal(s["input_ids"], row["input_ids"][0])
    pos, _ = model.get_rope_index(
        row["input_ids"],
        row["mm_token_type_ids"],
        image_grid_thw=row["image_grid_thw"],
        attention_mask=row["attention_mask"],
    )
    hf = pos[:, 0, :].T.cpu()  # (S, 3)
    real = row["attention_mask"][0].bool()
    tt = s["tt_mrope"].to(hf.dtype)
    diff = (hf[real] - tt[real]).abs().max().item()
    print(
        f"row {i}: {int(real.sum())} real tokens, image grid {row['image_grid_thw'].tolist()}, "
        f"max |HF - TorchTitan| position difference {diff}"
    )
    assert diff == 0, f"row {i}: positions differ"
print("OK: M-RoPE positions identical on every real token")
