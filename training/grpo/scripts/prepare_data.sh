#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# Downloads the retool training and AIME eval data into data/. Assumes you're cd'd into training/grpo/.
set -euo pipefail

hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir data/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir data/aime-2024

# aime-2024 has bare problems, but the scorer only accepts \boxed{} answers. Ask for one with the prefix and suffix
# that every dapo-math-17k prompt has. Prompts that already have them are left as they are.
python3 - data/aime-2024/aime-2024.jsonl <<'PY'
import json
import os
import sys

PREFIX = ("Solve the following math problem step by step. The last line of your response should be of the form "
          "Answer: \\boxed{$Answer} where $Answer is the answer to the problem.\n\n")
SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'

path = sys.argv[1]
with open(path) as f:
    rows = [json.loads(line) for line in f if line.strip()]
for row in rows:
    message = row["prompt"][0]
    if not message["content"].startswith(PREFIX):
        message["content"] = PREFIX + message["content"] + SUFFIX
with open(path + ".tmp", "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
os.replace(path + ".tmp", path)
PY
