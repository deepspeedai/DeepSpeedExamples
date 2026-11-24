# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Every flag the docs pass to train_vl.py, train.py or a recipe exists in that
script's parser, in fenced code blocks and in inline code alike. Inline code
that starts with a flag (`--cuts balanced`) must be a flag of either script."""

import glob
import importlib.util
import os
import re

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DOCS = [os.path.join(ROOT, "README.md")] + sorted(glob.glob(os.path.join(ROOT, "docs", "*.md")))
_SCRIPT = re.compile(r"(train_vl\.py|train\.py|recipes/[\w.-]+\.sh)")
_FLAG = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")


def _parser(script: str):
    spec = importlib.util.spec_from_file_location(script[:-3], os.path.join(ROOT, script))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_parser()


def _script_of(target: str) -> str:
    """The Python script a doc command runs: itself, or the one a recipe calls."""
    if not target.startswith("recipes/"):
        return target
    with open(os.path.join(ROOT, target)) as f:
        return _SCRIPT.search(f.read().split("exec ", 1)[1]).group(1)


def _commands(text: str):
    """Commands in fenced blocks (continuation lines joined) and inline code."""
    for block in re.findall(r"```[^\n]*\n(.*?)```", text, re.S):
        yield from block.replace("\\\n", " ").splitlines()
    yield from re.findall(r"(?<!`)`([^`\n]+)`(?!`)", re.sub(r"```.*?```", "", text, flags=re.S))


def documented_flags():
    """(doc, script, flag) for every flag written after a script on one command."""
    found = set()
    for doc in DOCS:
        with open(doc) as f:
            text = f.read()
        for command in _commands(text):
            match = _SCRIPT.search(command)
            if match is None:
                continue
            rest = command[match.end() :].split("#", 1)[0]
            for flag in _FLAG.findall(rest):
                found.add((os.path.basename(doc), _script_of(match.group(1)), flag))
    return sorted(found)


def bare_flags():
    """(doc, flag) for inline code that starts with a flag and names no script."""
    found = set()
    for doc in DOCS:
        with open(doc) as f:
            text = re.sub(r"```.*?```", "", f.read(), flags=re.S)
        for span in re.findall(r"(?<!`)`(--[^`\n]+)`(?!`)", text):
            if _SCRIPT.search(span) is None:
                found.update((os.path.basename(doc), flag) for flag in _FLAG.findall(span))
    return sorted(found)


def test_docs_document_flags():
    assert len(documented_flags()) >= 5
    assert len(bare_flags()) >= 10


@pytest.mark.parametrize("doc,script,flag", documented_flags())
def test_every_documented_flag_exists(doc, script, flag):
    options = set(_parser(script)._option_string_actions)
    assert flag in options, f"{doc}: {script} has no {flag}"


@pytest.mark.parametrize("doc,flag", bare_flags())
def test_every_inline_flag_exists(doc, flag):
    options = set(_parser("train_vl.py")._option_string_actions) | set(
        _parser("train.py")._option_string_actions
    )
    assert flag in options, f"{doc}: neither train_vl.py nor train.py has {flag}"
