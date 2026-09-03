"""Coverage tests for multilingual and code sandbox evaluators."""

from __future__ import annotations

from promptdiff.core.models import RunResult, TestCase
from promptdiff.evaluators.code_sandbox import (
    SafeCodeSandboxEvaluator,
    extract_python_blocks,
)
from promptdiff.evaluators.multilingual import MultilingualConsistencyEvaluator


def test_extract_python_blocks() -> None:
    text = "Here is the code:\n```python\nx = 1\ny = 2\n```\nAnd another:\n```\nz = 3\n```"
    blocks = extract_python_blocks(text)
    assert len(blocks) == 2
    assert "x = 1" in blocks[0]
    assert "z = 3" in blocks[1]


def test_code_sandbox_success() -> None:
    evaluator = SafeCodeSandboxEvaluator()
    code = "def add(a, b):\n    return a + b"
    test_code = "assert add(2, 3) == 5"
    result = evaluator.execute_snippet(code, test_code)

    assert result.executed is True
    assert result.passed is True
    assert result.error_message is None


def test_code_sandbox_assertion_failure() -> None:
    evaluator = SafeCodeSandboxEvaluator()
    code = "def add(a, b):\n    return a - b"
    test_code = "assert add(2, 3) == 5"
    result = evaluator.execute_snippet(code, test_code)

    assert result.executed is True
    assert result.passed is False
    assert "AssertionError" in (result.error_message or "")


def test_code_sandbox_evaluator_sync_and_async() -> None:
    evaluator = SafeCodeSandboxEvaluator()
    tc = TestCase(
        id="tc1",
        vars={"test_code": "assert double(4) == 8"},
    )
    r1 = RunResult(
        prompt_name="v1",
        test_case_id="tc1",
        rendered_prompt="def double",
        output="```python\ndef double(x):\n    return x * 2\n```",
        latency_ms=50.0,
        prompt_tokens=5,
        completion_tokens=10,
        total_tokens=15,
        cost_usd=0.0001,
        model="mock",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="tc1",
        rendered_prompt="def double",
        output="```python\ndef double(x):\n    return x * 3\n```",
        latency_ms=55.0,
        prompt_tokens=5,
        completion_tokens=10,
        total_tokens=15,
        cost_usd=0.0001,
        model="mock",
    )

    score = evaluator.evaluate(r1, r2, tc)
    assert score.v1_score == 1.0
    assert score.v2_score == 0.0
    assert score.passed is False


def test_multilingual_evaluator_languages() -> None:
    evaluator = MultilingualConsistencyEvaluator(min_parity_threshold=0.60)

    # Turkish
    s_tr = evaluator.evaluate_language_invariance("Bu bir test için yazılmış Türkçe cümledir.", "tr")
    assert s_tr > 0.5

    # German
    s_de = evaluator.evaluate_language_invariance("Das ist ein Test und das ist nicht schwer.", "de")
    assert s_de > 0.5

    # English
    s_en = evaluator.evaluate_language_invariance("This is the test with and for that purpose.", "en")
    assert s_en > 0.5

    # Empty text
    assert evaluator.evaluate_language_invariance("", "en") == 0.0

    # Unmapped language
    assert evaluator.evaluate_language_invariance("Some text", "zz") == 0.85


def test_multilingual_evaluator_sync() -> None:
    evaluator = MultilingualConsistencyEvaluator(min_parity_threshold=0.50)
    tc = TestCase(id="tc_lang", vars={"language": "de"})
    r1 = RunResult(
        prompt_name="v1",
        test_case_id="tc_lang",
        rendered_prompt="p",
        output="Das ist der beste Weg und nicht falsch.",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="tc_lang",
        rendered_prompt="p",
        output="Und das ist auch gut so.",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )
    score = evaluator.evaluate(r1, r2, tc)
    assert score.name == "multilingual"
    assert score.passed is True
