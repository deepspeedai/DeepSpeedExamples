# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Math prompts with reference answers, read from JSONL.

Each line holds a prompt (a chat message list, or a string taken as the
user message) and its label, as in the ``dapo-math-17k`` and
``aime-2024`` files. Rows keep their line number as ``index``, which the
trainer uses as the GRPO group index.
"""

import json

from torch.utils.data import Dataset


def load_prompts(path: str, prompt_key: str = "prompt", label_key: str = "label") -> list:
    rows = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record[prompt_key]
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]
            rows.append({"index": len(rows), "prompt": prompt, "label": str(record[label_key])})
    return rows


class PromptDataset(Dataset):

    def __init__(self, path: str, prompt_key: str = "prompt", label_key: str = "label"):
        self.rows = load_prompts(path, prompt_key, label_key)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def collate_rows(rows: list) -> list:
    """Keep a batch as a list of rows; prompts are rendered by the rollout engine."""
    return rows
