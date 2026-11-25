# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Importing the package must not import ray, deepspeed or cupy.

They are runtime dependencies imported lazily by the coordinator and stage
actors; import-time independence keeps the compile path and its tests
runnable on a CPU-only machine."""

import subprocess
import sys

SNIPPET = (
    "import sys; import ray_deepspeed_pipeline; "
    "leaked = [m for m in ('ray', 'deepspeed', 'cupy') if m in sys.modules]; "
    "assert not leaked, f'package import pulled in {leaked}'"
)


def test_package_import_does_not_import_ray_or_deepspeed():
    result = subprocess.run([sys.executable, "-c", SNIPPET], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
