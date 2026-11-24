# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Dependency directions, checked mechanically.

- Importing the package or any compile-path module pulls in none of ray,
  deepspeed or cupy: they are lazy runtime dependencies of the stage actors.
- The plan/config/protocol schema modules are plain data (no torch either),
  so the ExecutionPlan stays canonically serializable anywhere.
"""

import subprocess
import sys

PKG = "ray_deepspeed_pipeline"


def _import_leaks(module, forbidden):
    snippet = (
        f"import sys, importlib; importlib.import_module('{module}'); "
        f"print(sorted(m for m in {forbidden!r} if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return eval(out.stdout.strip())


def test_compile_path_never_imports_ray_or_deepspeed():
    for module in (
        PKG,
        f"{PKG}.compiler",
        f"{PKG}.partition",
        f"{PKG}.plan",
        f"{PKG}.config",
        f"{PKG}.protocols",
        f"{PKG}.support_matrix",
    ):
        assert _import_leaks(module, ["ray", "deepspeed", "cupy"]) == [], module


def test_schema_modules_are_plain_data():
    # checked via each module's own import statements (importing a submodule
    # always executes the package __init__, which may legitimately use torch)
    import ast
    import importlib.util

    for module in (f"{PKG}.plan", f"{PKG}.config", f"{PKG}.protocols", f"{PKG}.support_matrix"):
        source = open(importlib.util.find_spec(module).origin).read()
        for node in ast.walk(ast.parse(source)):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            for name in names:
                root = name.split(".")[0]
                assert root not in ("torch", "ray", "deepspeed"), (
                    f"{module} imports {name}: schema modules are plain data"
                )
