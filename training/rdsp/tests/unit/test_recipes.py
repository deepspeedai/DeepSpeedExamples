# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Every recipe in recipes/ is a valid command: its flags parse with the
script it calls, and its header states the hardware and the measured result."""

import importlib.util
import os
import shlex

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RECIPES = (
    sorted(f for f in os.listdir(os.path.join(ROOT, "recipes")) if f.endswith(".sh"))
    if os.path.isdir(os.path.join(ROOT, "recipes"))
    else []
)


def _parser(script: str):
    spec = importlib.util.spec_from_file_location(script[:-3], os.path.join(ROOT, script))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_parser()


def _command(text: str) -> tuple[str, list[str]]:
    """The script a recipe runs and the flags it passes (up to "$@")."""
    tokens = shlex.split(text.replace("\\\n", " "), comments=True)
    script = next(t for t in tokens if t.endswith(("train_vl.py", "train.py")))
    args = tokens[tokens.index(script) + 1 :]
    return os.path.basename(script), args[: args.index("$@")] if "$@" in args else args


def test_there_is_a_recipe_per_validated_layout():
    assert RECIPES == [
        "qwen3-text-pp.sh",
        "qwen35-2b-tp2pp2dp2.sh",
        "qwen35-2b-vision1-tp2dp2.sh",
        "qwen35-4b-coloc-tp2pp2dp2.sh",
        "qwen35-4b-tp2pp2dp2.sh",
        "qwen35-4b-tp4pp2dp1.sh",
        "qwen35-4b-vision1-tp2dp2.sh",
        "qwen35-4b-vision1-tp2pp2.sh",
        "qwen35-4b-vision1-tp4.sh",
        "qwen35-4b-vision2-tp2dp2.sh",
    ]


@pytest.mark.parametrize("recipe", RECIPES)
def test_every_recipe_parses(recipe):
    text = open(os.path.join(ROOT, "recipes", recipe)).read()
    script, args = _command(text)
    _parser(script).parse_args(args)
    assert "GPUs:" in text and "Measured:" in text
