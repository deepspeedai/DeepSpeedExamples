# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""CPU check: with data parallelism 2 (layout C3), TorchTitan's loader on rank r gives
microbatch k = exported row 2k + r, so each step trains on the same 64 consecutive rows
as Megatron-Bridge (global microbatch k = rows 2k, 2k + 1). Also builds the C3 configs.

    PYTHONPATH=rdsp/bench/torchtitan EXPORTED=<export dir> HF_ASSETS=<snapshot> \\
        <torchtitan venv>/bin/python rdsp/bench/torchtitan/checks/tt_dp_check.py
"""

import glob
import os

import cord_data
import cord_qwen35
import torch
from torchtitan.components.tokenizer import MultiModalTokenizer

export, hf = os.environ["EXPORTED"], os.environ["HF_ASSETS"]
SPECIAL = {
    "image_token": "<|image_pad|>",
    "video_token": "<|video_pad|>",
    "vision_start_token": "<|vision_start|>",
    "vision_end_token": "<|vision_end|>",
    "pad_token": "<|endoftext|>",
}
tokenizer = MultiModalTokenizer.Config(**SPECIAL).build(tokenizer_path=hf)
rows = [
    r
    for p in sorted(glob.glob(os.path.join(export, "step_*.pt")))
    for r in torch.load(p, weights_only=False)
]
for rank in (0, 1):
    loader = cord_data.exported_dataloader(export, data_parallel=2).build(
        dp_world_size=2,
        dp_rank=rank,
        tokenizer=tokenizer,
        max_context_length=2048,
        num_tokens_per_microbatch=2048,
    )
    it = iter(loader)
    for k in range(len(rows) // 2):
        mb = next(it)
        assert torch.equal(mb.input.flatten(), rows[2 * k + rank]["input_ids"][0]), (rank, k)
    print(
        f"rank {rank}: microbatches 0-{len(rows) // 2 - 1} = exported rows "
        f"{rank}, {rank + 2}, ..., {len(rows) - 2 + rank}"
    )
for name in ("qwen35_27b_cord_tp4dp2", "qwen35_27b_cord_tp4dp2_rc"):
    c = getattr(cord_qwen35, name)()
    p, t = c.parallelism, c.training
    accumulation = t.num_tokens_per_train_step // (t.num_tokens_per_microbatch_per_dp_rank * 2)
    recompute = type(c.activation_checkpoint).__name__ if c.activation_checkpoint else None
    print(
        f"{name}: dp_shard {p.data_parallel_shard_degree} TP {p.tensor_parallel_degree} "
        f"PP {p.pipeline_parallel_degree} SP {p.enable_sequence_parallel} "
        f"| tokens/microbatch {t.num_tokens_per_microbatch_per_dp_rank} "
        f"tokens/step {t.num_tokens_per_train_step} -> accumulation {accumulation} "
        f"| recompute {recompute} "
        f"| loader repeat {c.dataloader.repeat} shuffle {c.dataloader.shuffle}"
    )
print("OK")
