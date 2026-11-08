# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""A Python code interpreter the model calls during rollout.

Code passes an import allowlist and a pattern filter, then runs under a
memory limit and a timeout, as an async subprocess so concurrent calls do not
block each other. The filter only screens obvious misuse; it is not a security
boundary.
"""

import ast
import asyncio
import os
import re
import sys
import tempfile
import traceback

from deepspeed.runtime.rollout import Tool

ALLOWED_MODULES = {
    "math",
    "random",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "operator",
    "statistics",
    "decimal",
    "fractions",
}

DANGEROUS_PATTERNS = [
    r"import\s+os",
    r"import\s+sys",
    r"import\s+subprocess",
    r"import\s+shutil",
    r"import\s+glob",
    r"import\s+pathlib",
    r"__import__",
    r"eval\s*\(",
    r"exec\s*\(",
    r"open\s*\(",
    r"file\s*\(",
    r"input\s*\(",
    r"raw_input\s*\(",
    r"compile\s*\(",
    r"execfile\s*\(",
    r"getattr\s*\(",
    r"setattr\s*\(",
    r"delattr\s*\(",
    r"hasattr\s*\(",
    r"globals\s*\(",
    r"locals\s*\(",
    r"vars\s*\(",
    r"dir\s*\(",
    r"type\s*\(",
    r"isinstance\s*\(",
    r"issubclass\s*\(",
    r"super\s*\(",
    r"property\s*\(",
    r"staticmethod\s*\(",
    r"classmethod\s*\(",
    r"__\w+__",
]

# The user code runs inside a try block so its output and errors come back in
# the format the model trains with.
WRAPPER = """import sys
import traceback
from io import StringIO
import resource

try:
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, -1))
except Exception:
    pass

old_stdout = sys.stdout
old_stderr = sys.stderr
stdout_capture = StringIO()
stderr_capture = StringIO()
sys.stdout = stdout_capture
sys.stderr = stderr_capture

try:
USER_CODE

    stdout_output = stdout_capture.getvalue()
    stderr_output = stderr_capture.getvalue()
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    result = ""
    if stdout_output:
        result += f"Output:\\n{stdout_output}"
    if stderr_output:
        result += f"\\nErrors:\\n{stderr_output}"
    print(result)
except Exception as e:
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    print(f"Error: {str(e)}\\nTraceback:\\n{traceback.format_exc()}")
"""


def imported_modules(tree: ast.AST) -> set:
    """The top-level modules a parsed program imports: ``a`` for ``import a.b`` and for ``from a.b import c``."""
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # A relative import has no module to check against the allowlist.
            modules.add(node.module.split(".")[0] if node.module and not node.level else ".")
    return modules


# A Markdown code fence around the whole program: ```py / ```python / ``` ... ```.
_CODE_FENCE = re.compile(r"\A\s*```[\w+-]*[ \t]*\n(.*?)\n?[ \t]*(?:```\s*)?\Z", re.DOTALL)


def strip_code_fence(code: str) -> str:
    """``code`` without a Markdown fence around it, as Qwen3-2507 models write their ``code_interpreter`` calls.

    Only a fence that wraps the whole argument is removed (its closing line may be missing); code that merely
    contains backticks is left as it is.
    """
    match = _CODE_FENCE.match(code)
    return match.group(1) if match else code


def check_code_safety(code: str):
    """Return an error message if ``code`` matches a blocked pattern, does not parse or imports a blocked module, else None."""
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            return f"Code contains dangerous pattern: {pattern}"
    try:
        tree = ast.parse(code, filename="<code>")
    except SyntaxError as e:
        # Python compiles the whole program before running any of it, so code that does not parse cannot run;
        # return the error Python would raise.
        return "Code does not compile\n" + "".join(traceback.format_exception_only(type(e), e)).rstrip()
    for module in sorted(imported_modules(tree)):
        if module not in ALLOWED_MODULES:
            return f"Import of '{module}' is not allowed"
    return None


class CodeInterpreter(Tool):
    name = "code_interpreter"
    # The tool returns only what the code prints, as a script does: a bare expression on the last line, as in a
    # notebook, gives no output.
    description = ("A tool for executing Python code in a safe sandbox environment. Only printed output is returned, "
                   "so print() every value you need.")
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute"
            }
        },
        "required": ["code"],
    }

    def __init__(self,
                 timeout: float = 120.0,
                 memory_limit_gb: int = 4,
                 max_concurrency: int = 32,
                 max_output_chars: int = 4096):
        self.timeout = timeout
        self.memory_limit = memory_limit_gb * 1024**3
        self.max_concurrency = max_concurrency
        self.max_output_chars = max_output_chars
        self._loop = None
        self._semaphore = None

    def _limiter(self) -> asyncio.Semaphore:
        # The rollout runs each turn's calls in a new event loop, and a
        # semaphore only works in the loop it was first used in.
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._loop = loop
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    async def execute(self, arguments) -> str:
        code = arguments.get("code", "")
        if isinstance(code, list):
            code = "\n".join(str(line) for line in code)
        if isinstance(code, str):
            code = strip_code_fence(code)
        output = await self._execute(code)
        if len(output) > self.max_output_chars:
            output = output[:self.max_output_chars] + "\n... (output truncated)"
        return output

    async def _execute(self, code) -> str:
        """The sandbox's output for ``code``, before truncation."""
        if not isinstance(code, str) or not code.strip():
            return "Error: No code provided"
        error = check_code_safety(code)
        if error is not None:
            return f"Error: {error}"
        async with self._limiter():
            return await self._run(code)

    async def _run(self, code: str) -> str:
        indented = "\n".join("    " + line for line in code.split("\n"))
        script = WRAPPER.replace("MEMORY_LIMIT", str(self.memory_limit)).replace("USER_CODE", indented)
        with tempfile.TemporaryDirectory(prefix="python_sandbox_") as workdir:
            path = os.path.join(workdir, "code.py")
            with open(path, "w") as f:
                f.write(script)
            env = dict(os.environ, PYTHONPATH=workdir, PYTHONUNBUFFERED="1")
            process = await asyncio.create_subprocess_exec(sys.executable,
                                                           path,
                                                           stdout=asyncio.subprocess.PIPE,
                                                           stderr=asyncio.subprocess.PIPE,
                                                           cwd=workdir,
                                                           env=env)
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return f"Error: Code execution timed out after {self.timeout:g} seconds"
        if process.returncode != 0:
            return f"Error: Process exited with code {process.returncode}\n{stderr.decode(errors='replace')}"
        return stdout.decode(errors="replace").strip()
