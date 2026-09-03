"""Tests for promptdiff pytest plugin and fixtures."""

from __future__ import annotations

import pytest

from promptdiff.core.models import TestCase

pytest_plugins = ["pytester"]


def test_pytest_plugin_fixture(promptdiff_eval) -> None:
    """Test pytest fixture provided by pytest_plugin."""
    report = promptdiff_eval(
        v1="Say hello: {{name}}",
        v2="Greet user: {{name}}",
        dataset=[TestCase(id="1", vars={"name": "Alice"})],
        mock=True,
    )
    assert report.total_cases == 1
    assert report.verdict.passed is True


@pytest.mark.asyncio
async def test_promptdiff_eval_in_async_test_raises_clear_error(promptdiff_eval) -> None:
    """Test that calling sync promptdiff_eval inside an async test raises descriptive RuntimeError."""
    with pytest.raises(RuntimeError, match="promptdiff_eval is a synchronous fixture and cannot be called"):
        promptdiff_eval(
            v1="v1 prompt",
            v2="v2 prompt",
            mock=True,
        )


def test_pytester_subprocess_integration(pytester: pytest.Pytester) -> None:
    """Integration test verifying entry-point registered fixtures in a real pytest subprocess."""
    pytester.makepyfile(
        """
        import pytest

        def test_sync_run(promptdiff_eval):
            report = promptdiff_eval(
                v1="Hello world {{name}}",
                v2="Hello universe {{name}}",
                mock=True,
            )
            assert report.total_cases >= 1
            assert report.verdict.passed is True

        @pytest.mark.asyncio
        async def test_async_run(prompt_diff):
            report = await prompt_diff.compare(
                v1="Hello world {{name}}",
                v2="Hello universe {{name}}",
                mock=True,
            )
            assert report.total_cases >= 1
            assert report.verdict.passed is True
        """
    )
    result = pytester.runpytest("-p", "promptdiff", "-o", "asyncio_mode=auto")
    result.assert_outcomes(passed=2)
