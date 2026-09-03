"""Security and containment verification tests for SafeCodeSandboxEvaluator."""

from __future__ import annotations

import pytest

from promptdiff.core.models import RunResult, TestCase
from promptdiff.evaluators.code_sandbox import SafeCodeSandboxEvaluator


def test_code_sandbox_timeout_infinite_loop() -> None:
    """Ensure infinite loops are terminated promptly and report TimeoutError."""
    evaluator = SafeCodeSandboxEvaluator(timeout_sec=0.5)
    infinite_loop_code = "while True:\n    pass"

    result = evaluator.execute_snippet(infinite_loop_code)
    assert result.passed is False
    assert result.executed is False
    assert "TimeoutError" in (result.error_message or "")


def test_code_sandbox_exploit_subclasses_blocked() -> None:
    """Ensure attempts to break out of sandbox via object.__subclasses__() fail."""
    evaluator = SafeCodeSandboxEvaluator(timeout_sec=2.0)

    # Attempt to locate _wrap_close or similar class and access its os.system
    exploit_code = """
subclasses = ().__class__.__bases__[0].__subclasses__()
wrap_close = [c for c in subclasses if c.__name__ == "_wrap_close"]
if wrap_close:
    fn = wrap_close[0].__init__.__globals__.get("system")
    if fn is not None:
        fn("echo pwned")
"""
    result = evaluator.execute_snippet(exploit_code)
    # The exploit should not succeed (either raises PermissionError or TypeError or fails)
    assert result.passed is False or result.error_message is not None or "PermissionError" in str(result.error_message)


def test_code_sandbox_filesystem_access_blocked() -> None:
    """Ensure file write and read attempts are blocked by sandbox permissions."""
    evaluator = SafeCodeSandboxEvaluator(timeout_sec=2.0)

    write_code = "with open('sandbox_escape_test.txt', 'w') as f:\n    f.write('hacked')"
    result = evaluator.execute_snippet(write_code)
    assert result.passed is False
    assert "PermissionError" in (result.error_message or "")


def test_code_sandbox_network_access_blocked() -> None:
    """Ensure network socket connections cannot be opened within sandbox."""
    evaluator = SafeCodeSandboxEvaluator(timeout_sec=2.0)

    # Attempt importing socket and creating connection
    socket_code = """
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('1.1.1.1', 80))
"""
    result = evaluator.execute_snippet(socket_code)
    assert result.passed is False
    assert "PermissionError" in (result.error_message or "")


def test_code_sandbox_dangerous_module_imports_blocked() -> None:
    """Ensure dangerous OS/subprocess modules cannot be imported."""
    evaluator = SafeCodeSandboxEvaluator(timeout_sec=2.0)

    for mod in ["os", "subprocess", "shutil", "ctypes"]:
        code = f"import {mod}"
        res = evaluator.execute_snippet(code)
        assert res.passed is False
        assert "PermissionError" in (res.error_message or "")


def test_code_sandbox_legitimate_code_succeeds() -> None:
    """Ensure legitimate mathematical and algorithmic code runs cleanly."""
    evaluator = SafeCodeSandboxEvaluator(timeout_sec=2.0)

    code = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""
    unit_test = "assert quicksort([3, 1, 4, 1, 5, 9, 2, 6]) == [1, 1, 2, 3, 4, 5, 6, 9]"
    res = evaluator.execute_snippet(code, unit_test)
    assert res.passed is True
    assert res.error_message is None


@pytest.mark.asyncio
async def test_code_sandbox_async_evaluate_timeout_and_security() -> None:
    """Test async_evaluate method with timeout handling."""
    evaluator = SafeCodeSandboxEvaluator(timeout_sec=0.5)

    tc = TestCase(id="sec_tc", vars={"test_code": "while True: pass"})
    r1 = RunResult(
        prompt_name="v1",
        test_case_id="sec_tc",
        rendered_prompt="test",
        output="def run(): return 1",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0001,
        model="mock",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="sec_tc",
        rendered_prompt="test",
        output="def run(): return 2",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0001,
        model="mock",
    )

    score = await evaluator.async_evaluate(r1, r2, tc)
    assert score.passed is False
    assert "TimeoutError" in score.message
