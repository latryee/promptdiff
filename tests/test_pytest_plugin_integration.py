"""Integration tests verifying pytest11 entry point and fixtures in isolated child subprocess."""

from __future__ import annotations

import pytest

pytest_plugins = ["pytester"]


def test_plugin_entrypoint_subprocess_sync_and_async(pytester: pytest.Pytester) -> None:
    """Verify entrypoint registered fixtures run seamlessly in isolated pytest subprocess."""
    pytester.makepyfile(
        """
        import pytest
        from promptdiff.core.models import TestCase

        def test_sync_fixture_evaluation(promptdiff_eval):
            report = promptdiff_eval(
                v1="Answer customer: {{query}}",
                v2="Answer customer concisely: {{query}}",
                dataset=[TestCase(id="1", vars={"query": "Pricing info"})],
                mock=True,
            )
            assert report.total_cases == 1
            assert report.verdict.passed is True

        @pytest.mark.asyncio
        async def test_async_fixture_evaluation(prompt_diff):
            report = await prompt_diff.compare(
                v1="Answer customer: {{query}}",
                v2="Answer customer concisely: {{query}}",
                dataset=[TestCase(id="1", vars={"query": "Billing info"})],
                mock=True,
            )
            assert report.total_cases == 1
            assert report.verdict.passed is True
        """
    )
    result = pytester.runpytest("-p", "promptdiff", "-o", "asyncio_mode=auto")
    result.assert_outcomes(passed=2)


def test_plugin_entrypoint_event_loop_guard(pytester: pytest.Pytester) -> None:
    """Verify sync promptdiff_eval inside running event loop raises informative RuntimeError."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.asyncio
        async def test_sync_in_async_raises_runtime_error(promptdiff_eval):
            with pytest.raises(RuntimeError, match="promptdiff_eval is a synchronous fixture"):
                promptdiff_eval(
                    v1="v1 prompt",
                    v2="v2 prompt",
                    mock=True,
                )
        """
    )
    result = pytester.runpytest("-p", "promptdiff", "-o", "asyncio_mode=auto")
    result.assert_outcomes(passed=1)
