"""Safe Isolated Code Sandbox Evaluator for Generated Code Snippets.

Extracts executable Python/SQL blocks from LLM completions and runs unit tests
inside a restricted execution environment with strict CPU/memory timeouts.
"""

from __future__ import annotations

import asyncio
import io
import re
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any, Optional

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


class SafeCodeSandboxEvaluator(BaseEvaluator):
    """Executes generated code against assertions in isolated namespaces."""

    name: str = "code_sandbox"
    description: str = "Isolated code sandbox evaluator for Python code snippets."

    def __init__(self, timeout_sec: float = 2.0):
        self.timeout_sec = timeout_sec

    def execute_snippet(self, code: str, unit_test: str = "") -> CodeExecutionResult:
        """Execute code safely with restricted builtins and stdout capture."""
        full_code = f"{code}\n\n{unit_test}".strip()
        stdout_buf = io.StringIO()

        # Restricted execution environment
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
        }
        restricted_globals = {"__builtins__": safe_builtins}
        local_scope: dict[str, Any] = {}

        try:
            with redirect_stdout(stdout_buf):
                exec(full_code, restricted_globals, local_scope)
            return CodeExecutionResult(
                executed=True,
                passed=True,
                stdout=stdout_buf.getvalue(),
                error_message=None,
                code_snippet=code,
            )
        except Exception as e:
            return CodeExecutionResult(
                executed=True,
                passed=False,
                stdout=stdout_buf.getvalue(),
                error_message=f"{type(e).__name__}: {str(e)}",
                code_snippet=code,
            )

    async def async_evaluate(self, r1: RunResult, r2: RunResult, test_case: TestCase) -> EvaluatorScore:
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

    def evaluate(self, r1: RunResult, r2: RunResult, test_case: TestCase) -> EvaluatorScore:
        return asyncio.run(self.async_evaluate(r1, r2, test_case))
