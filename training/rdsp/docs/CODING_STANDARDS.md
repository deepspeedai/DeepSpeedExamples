# Coding standards

Short on purpose. Where these conflict with a documented design decision in
`docs/ENGINEERING.md`, the design decision wins.

## Comments and docstrings

- A docstring states what the function does and anything a caller must know
  that the signature doesn't say: invariants, ordering, failure modes. One
  line when that is enough.
- A comment explains *why*, never *what*. Delete comments that restate the
  code.
- Keep the reasons behind non-obvious correctness choices (gradient scaling,
  DeepSpeed workarounds, failure semantics), stated in as few words as work.
- No project-history references in code: no phase labels (`P5`, `M2`), no
  plan section numbers (`Impl §9`), no "was / used to / now". History
  belongs in `docs/` and git.
- No commented-out code.

## Structure

- Modules are deep: a small interface over the real work. Private helpers
  start with `_`.
- `import ray_deepspeed_pipeline` must not import `ray`, `deepspeed` or
  `cupy` (enforced by `tests/architecture`). Import them lazily at the call
  site.
- Plain data (`plan.py`, `checkpoint.py`, `boundary.py`, `schedule.py`)
  stays free of torch/ray/deepspeed imports, except `boundary.py`, which
  needs torch for slicing.
- Errors raised to users are the types in `errors.py`, with a message that
  says what went wrong and what to do.

## Names

- Names say what a thing is (`microbatch`, `stage`, `rank`, `grid`), not how
  it is stored. Single letters only for loop indices and short math.

## Tests

- Test through the public or module interface, not private helpers.
- CPU tests must pass with `pytest -q` before any change is done. GPU tests
  run on Modal through `scripts/modal_tests.py`, and cost money: run them
  only when needed.

## Tooling

- `ruff check .` must pass (config in `pyproject.toml`).
