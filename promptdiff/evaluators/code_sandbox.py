"""Safe Isolated Code Sandbox Evaluator for Generated Code Snippets.

Extracts executable Python blocks from LLM completions and runs unit tests
inside an isolated subprocess execution environment with strict timeouts,
resource limits, and exploit protection.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess  # nosec: B404 # subprocess module import is required for sandbox execution isolation
import sys
from dataclasses import dataclass
from typing import Optional

from promptdiff.core.models import EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.base import BaseEvaluator


@dataclass
class CodeExecutionResult:
    """Outcome of sandboxed execution."""

    executed: bool
    passed: bool
    stdout: str
    error_message: Optional[str]
    code_snippet: str


def extract_python_blocks(text: str) -> list[str]:
    """Extract Python code blocks fenced with ```python ... ```."""
    pattern = re.compile(r"```(?:python|py)?\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)
    return [m.strip() for m in matches if m.strip()]


_SANDBOX_RUNNER_SCRIPT = r"""
import sys
import json
import io
from contextlib import redirect_stdout, redirect_stderr

# 1. Apply platform resource limits if available (POSIX)
try:
    import resource
    timeout_sec = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    mem_limit_mb = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    cpu_limit = max(1, int(timeout_sec) + 1)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
    resource.setrlimit(resource.RLIMIT_AS, (mem_limit_mb * 1024 * 1024, mem_limit_mb * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
except Exception:
    pass

# 2. Block dangerous modules from being imported
BLOCKED_MODULES = {
    "os", "subprocess", "shutil", "socket", "ctypes", "pathlib",
    "http", "urllib", "requests", "httpx", "aiohttp", "multiprocessing",
    "threading", "webbrowser", "ftplib", "smtplib", "telnetlib", "posix", "nt"
}

def _deny_permission(name):
    raise PermissionError(f"Access to '{name}' is prohibited in sandbox.")

class SandboxImportBlocker:
    def find_spec(self, fullname, path, target=None):
        root = fullname.split(".")[0]
        if root in BLOCKED_MODULES:
            raise PermissionError(f"Importing '{fullname}' is prohibited in sandbox.")
        return None

sys.meta_path.insert(0, SandboxImportBlocker())

# Purge any dangerous modules already loaded
for mod in list(sys.modules.keys()):
    if mod.split(".")[0] in BLOCKED_MODULES:
        try:
            del sys.modules[mod]
        except Exception:
            pass

# 3. Neutralize dangerous callables in loaded subclasses to prevent object.__subclasses__ escape
def _blocked_introspection(*args, **kwargs):
    raise PermissionError("Access to dangerous function via subclass introspection is prohibited in sandbox.")

DANGEROUS_ATTRIBUTES = {
    "system", "popen", "spawn", "spawnl", "spawnle", "spawnv", "spawnve",
    "execv", "execve", "fork", "kill", "remove", "unlink", "rmdir", "rename"
}

try:
    for cls in ().__class__.__bases__[0].__subclasses__():
        init = getattr(cls, "__init__", None)
        if init and hasattr(init, "__globals__"):
            for k in list(init.__globals__.keys()):
                if k in DANGEROUS_ATTRIBUTES:
                    init.__globals__[k] = _blocked_introspection
except Exception:
    pass

# 4. Block socket network operations
try:
    import socket
    socket.socket = _blocked_introspection  # type: ignore
    socket.create_connection = _blocked_introspection  # type: ignore
    socket.getaddrinfo = _blocked_introspection  # type: ignore
except Exception:
    pass

def _blocked_open(*args, **kwargs):
    raise PermissionError("Filesystem access is prohibited in sandbox.")

safe_builtins = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "AssertionError": AssertionError,
    "ZeroDivisionError": ZeroDivisionError,
    "AttributeError": AttributeError,
    "OverflowError": OverflowError,
    "StopIteration": StopIteration,
    "RuntimeError": RuntimeError,
    "PermissionError": PermissionError,
    "open": _blocked_open,
    "__import__": __import__,
}

payload = json.loads(sys.stdin.read())
full_code = payload["full_code"]

stdout_buf = io.StringIO()
stderr_buf = io.StringIO()
execution_scope = {"__builtins__": safe_builtins, "__name__": "__main__"}

try:
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        exec(full_code, execution_scope, execution_scope)
    out = stdout_buf.getvalue()
    result = {
        "executed": True,
        "passed": True,
        "stdout": out,
        "error_message": None,
    }
except Exception as e:
    out = stdout_buf.getvalue()
    result = {
        "executed": True,
        "passed": False,
        "stdout": out,
        "error_message": f"{type(e).__name__}: {str(e)}",
    }

sys.stdout.write(json.dumps(result))
"""


class SafeCodeSandboxEvaluator(BaseEvaluator):
    """Executes generated code against assertions in isolated subprocesses."""

    name: str = "code_sandbox"
    description: str = "Isolated code sandbox evaluator for Python code snippets."

    def __init__(self, timeout_sec: float = 2.0, memory_limit_mb: int = 256):
        self.timeout_sec = timeout_sec
        self.memory_limit_mb = memory_limit_mb

    def execute_snippet(self, code: str, unit_test: str = "") -> CodeExecutionResult:
        """Execute code safely inside an isolated Python subprocess with resource limits and security guards."""
        full_code = f"{code}\n\n{unit_test}".strip()
        payload = json.dumps({"full_code": full_code, "code": code})

        cmd = [
            sys.executable,
            "-I",
            "-s",
            "-B",
            "-c",
            _SANDBOX_RUNNER_SCRIPT,
            str(self.timeout_sec),
            str(self.memory_limit_mb),
        ]

        try:
            proc = subprocess.run(  # nosec: B603 # subprocess execution uses explicit sys.executable with isolated flags and no shell
                cmd,
                input=payload,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return CodeExecutionResult(
                executed=False,
                passed=False,
                stdout="",
                error_message=f"TimeoutError: Execution exceeded timeout limit of {self.timeout_sec}s",
                code_snippet=code,
            )
        except Exception as err:
            return CodeExecutionResult(
                executed=False,
                passed=False,
                stdout="",
                error_message=f"{type(err).__name__}: {str(err)}",
                code_snippet=code,
            )

        if proc.returncode != 0 and not proc.stdout.strip():
            error_msg = proc.stderr.strip() or f"Process terminated with exit code {proc.returncode}"
            return CodeExecutionResult(
                executed=False,
                passed=False,
                stdout="",
                error_message=error_msg,
                code_snippet=code,
            )

        try:
            res_data = json.loads(proc.stdout.strip())
            return CodeExecutionResult(
                executed=bool(res_data.get("executed", True)),
                passed=bool(res_data.get("passed", False)),
                stdout=str(res_data.get("stdout", "")),
                error_message=res_data.get("error_message"),
                code_snippet=code,
            )
        except Exception:
            error_msg = (
                proc.stderr.strip() or proc.stdout.strip() or f"Invalid response from sandbox (code {proc.returncode})"
            )
            return CodeExecutionResult(
                executed=False,
                passed=False,
                stdout=proc.stdout,
                error_message=error_msg,
                code_snippet=code,
            )

    async def async_evaluate(self, r1: RunResult, r2: RunResult, test_case: TestCase) -> EvaluatorScore:
        test_assertion = test_case.vars.get("test_code", "") or test_case.vars.get("assert_code", "")
        blocks_v1 = extract_python_blocks(r1.output)
        blocks_v2 = extract_python_blocks(r2.output)

        code_v1 = blocks_v1[0] if blocks_v1 else r1.output
        code_v2 = blocks_v2[0] if blocks_v2 else r2.output

        loop = asyncio.get_running_loop()
        res1 = await loop.run_in_executor(None, self.execute_snippet, code_v1, test_assertion)
        res2 = await loop.run_in_executor(None, self.execute_snippet, code_v2, test_assertion)

        v1_score = 1.0 if res1.passed else 0.0
        v2_score = 1.0 if res2.passed else 0.0

        msg = "Code passed sandbox verification." if res2.passed else f"Sandbox failure: {res2.error_message}"

        return EvaluatorScore(
            name="code_sandbox",
            passed=res2.passed,
            v1_score=v1_score,
            v2_score=v2_score,
            delta=v2_score - v1_score,
            delta_pct=round((v2_score - v1_score) * 100.0, 1),
            message=msg,
        )

    def evaluate(self, r1: RunResult, r2: RunResult, test_case: TestCase) -> EvaluatorScore:
        test_assertion = test_case.vars.get("test_code", "") or test_case.vars.get("assert_code", "")
        blocks_v1 = extract_python_blocks(r1.output)
        blocks_v2 = extract_python_blocks(r2.output)

        code_v1 = blocks_v1[0] if blocks_v1 else r1.output
        code_v2 = blocks_v2[0] if blocks_v2 else r2.output

        res1 = self.execute_snippet(code_v1, test_assertion)
        res2 = self.execute_snippet(code_v2, test_assertion)

        v1_score = 1.0 if res1.passed else 0.0
        v2_score = 1.0 if res2.passed else 0.0

        msg = "Code passed sandbox verification." if res2.passed else f"Sandbox failure: {res2.error_message}"

        return EvaluatorScore(
            name="code_sandbox",
            passed=res2.passed,
            v1_score=v1_score,
            v2_score=v2_score,
            delta=v2_score - v1_score,
            delta_pct=round((v2_score - v1_score) * 100.0, 1),
            message=msg,
        )
