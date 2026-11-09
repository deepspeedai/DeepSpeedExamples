# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Retool reward: DAPO strict-box answer check.

The answer check is adapted from verl and EleutherAI's
lm-evaluation-harness under the Apache License 2.0:
Copyright 2024 Bytedance Ltd. and/or its affiliates, Copyright 2022
EleutherAI and the HuggingFace Inc. team.
"""

from typing import Optional


def last_boxed_only_string(text: str) -> Optional[str]:
    """The last ``\\boxed{...}`` expression in ``text``, braces balanced, or None."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth = 0
    for end in range(start, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                return text[start:end + 1]
    return None


def remove_boxed(boxed: str) -> str:
    return boxed[len("\\boxed{"):-1]


def compute_score(solution: str, label: str) -> dict:
    """+1 if the last boxed answer near the end of ``solution`` equals ``label``, else -1.

    Only the last 100 characters are searched, so the answer must close the
    response, as the DAPO prompt asks.
    """
    boxed = last_boxed_only_string(solution[-100:])
    pred = remove_boxed(boxed) if boxed is not None else None
    correct = pred == label
    return {"score": 1.0 if correct else -1.0, "acc": correct, "pred": pred or ""}


def retool_reward(solution: str, label: str, num_tool_calls: int, tool_call_bonus: bool = False) -> dict:
    """Retool reward: the strict-box score, plus an optional tool bonus.

    The bonus lifts a wrong answer by 0.05 per tool call, up to -0.6, so it
    never reaches a correct one.
    """
    result = compute_score(solution, label)
    if tool_call_bonus and result["score"] < 0:
        result["score"] = min(-0.6, result["score"] + num_tool_calls / 2 * 0.1)
    return result
