"""Pytest Plugin for promptdiff (pytest-promptdiff).

Enables developers to execute and assert prompt regressions directly inside pytest test suites
using the `@pytest.mark.promptdiff` decorator or `promptdiff_eval` fixture.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

import pytest

if TYPE_CHECKING:
    from promptdiff.core.models import DiffReport


def pytest_configure(config: Any) -> None:
    """Register promptdiff custom marker in pytest."""
    config.addinivalue_line(
        "markers",
        "promptdiff(v1, v2, dataset, model='gpt-4o', eval='json_validity,latency,cost,similarity', assert_rules=None, mock=True): "
        "Run automated prompt regression testing between two prompt versions.",
    )


class PromptDiffPluginHelper:
    """Pytest test fixture helper providing both async and sync prompt regression comparisons."""

    async def compare(
        self,
        v1: str,
        v2: str,
        test_cases: Any = None,
        dataset: Any = None,
        model: str = "gpt-4o",
        model_v1: Optional[str] = None,
        model_v2: Optional[str] = None,
        eval_metrics: str = "json_validity,latency,cost,similarity,llm_judge",
        assert_rules: Optional[list[str]] = None,
        mock: bool = True,
        concurrency: int = 4,
    ) -> DiffReport:
        """Run asynchronous regression comparison (as demonstrated in README)."""
        from promptdiff.sdk import async_compare as sdk_async_compare

        cases = test_cases if test_cases is not None else dataset
        return await sdk_async_compare(
            v1=v1,
            v2=v2,
            dataset=cases,
            model=model,
            model_v1=model_v1,
            model_v2=model_v2,
            eval_metrics=eval_metrics,
            assertions=assert_rules,
            mock=mock,
            concurrency=concurrency,
        )

    async def async_compare(
        self,
        v1: str,
        v2: str,
        test_cases: Any = None,
        dataset: Any = None,
        model: str = "gpt-4o",
        model_v1: Optional[str] = None,
        model_v2: Optional[str] = None,
        eval_metrics: str = "json_validity,latency,cost,similarity,llm_judge",
        assert_rules: Optional[list[str]] = None,
        mock: bool = True,
        concurrency: int = 4,
    ) -> DiffReport:
        """Explicit async regression comparison."""
        return await self.compare(
            v1=v1,
            v2=v2,
            test_cases=test_cases,
            dataset=dataset,
            model=model,
            model_v1=model_v1,
            model_v2=model_v2,
            eval_metrics=eval_metrics,
            assert_rules=assert_rules,
            mock=mock,
            concurrency=concurrency,
        )

    def sync_compare(
        self,
        v1: str,
        v2: str,
        test_cases: Any = None,
        dataset: Any = None,
        model: str = "gpt-4o",
        model_v1: Optional[str] = None,
        model_v2: Optional[str] = None,
        eval_metrics: str = "json_validity,latency,cost,similarity,llm_judge",
        assert_rules: Optional[list[str]] = None,
        mock: bool = True,
        concurrency: int = 4,
    ) -> DiffReport:
        """Synchronous prompt comparison."""
        from promptdiff.sdk import compare as sdk_compare

        cases = test_cases if test_cases is not None else dataset
        return sdk_compare(
            v1=v1,
            v2=v2,
            dataset=cases,
            model=model,
            model_v1=model_v1,
            model_v2=model_v2,
            eval_metrics=eval_metrics,
            assertions=assert_rules,
            mock=mock,
            concurrency=concurrency,
        )


@pytest.fixture
def prompt_diff() -> PromptDiffPluginHelper:
    """Pytest fixture providing `prompt_diff` helper with `await prompt_diff.compare(...)`."""
    return PromptDiffPluginHelper()


@pytest.fixture
def promptdiff_eval() -> Callable[..., DiffReport]:
    """Pytest fixture providing synchronous prompt evaluation helper."""

    def _eval(
        v1: str,
        v2: str,
        dataset: Optional[str] = None,
        model: str = "gpt-4o",
        model_v1: Optional[str] = None,
        model_v2: Optional[str] = None,
        eval_metrics: str = "json_validity,latency,cost,similarity,llm_judge",
        assert_rules: Optional[list[str]] = None,
        mock: bool = True,
        concurrency: int = 4,
    ) -> DiffReport:
        m1 = model_v1 or model
        m2 = model_v2 or model

        from promptdiff.core.config import load_prompt_file
        from promptdiff.core.runner import PromptDiffRunner
        from promptdiff.evaluators.registry import get_evaluators
        from promptdiff.providers.registry import get_provider
        from promptdiff.sdk import _resolve_testcases

        p1 = load_prompt_file(v1, version_name="v1", model=m1)
        p2 = load_prompt_file(v2, version_name="v2", model=m2)

        test_cases = _resolve_testcases(dataset)

        prov1 = get_provider(model_name=m1, force_mock=mock)
        prov2 = get_provider(model_name=m2, force_mock=mock)
        evaluators = get_evaluators([eval_metrics])

        runner = PromptDiffRunner(
            v1_prompt=p1,
            v2_prompt=p2,
            provider_v1=prov1,
            provider_v2=prov2,
            evaluators=evaluators,
            assertions=assert_rules,
            concurrency=concurrency,
        )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            try:
                import nest_asyncio

                nest_asyncio.apply(loop)
                return loop.run_until_complete(runner.run(test_cases))
            except ImportError:
                raise RuntimeError(
                    "promptdiff_eval is a synchronous fixture and cannot be called from an already running event loop "
                    "(e.g., inside an async test). Use the async 'prompt_diff' fixture with "
                    "'await prompt_diff.compare(...)' instead, or install 'nest_asyncio'."
                )

        return asyncio.run(runner.run(test_cases))

    return _eval


@pytest.fixture
def report(request: Any) -> Any:
    """Fixture providing access to DiffReport generated by @pytest.mark.promptdiff."""
    rep = getattr(request.node, "_promptdiff_report", None)
    if rep is None:
        raise RuntimeError("The 'report' fixture is only available on tests marked with @pytest.mark.promptdiff")
    return rep


@pytest.fixture
def diff_report(request: Any) -> Any:
    """Alias fixture providing access to DiffReport generated by @pytest.mark.promptdiff."""
    return report(request)


def pytest_runtest_setup(item: Any) -> None:
    """Execute regression comparison during test setup if marked with @pytest.mark.promptdiff."""
    marker = item.get_closest_marker("promptdiff")
    if marker is None:
        return

    v1 = marker.kwargs.get("v1") or (marker.args[0] if len(marker.args) > 0 else None)
    v2 = marker.kwargs.get("v2") or (marker.args[1] if len(marker.args) > 1 else None)
    dataset = marker.kwargs.get("dataset") or (marker.args[2] if len(marker.args) > 2 else None)
    model = marker.kwargs.get("model", "gpt-4o")
    model_v1 = marker.kwargs.get("model_v1")
    model_v2 = marker.kwargs.get("model_v2")
    eval_metrics = marker.kwargs.get("eval", marker.kwargs.get("eval_metrics", "json_validity,latency,cost,similarity"))
    assert_rules = marker.kwargs.get("assert_rules", marker.kwargs.get("assertions", None))
    mock = marker.kwargs.get("mock", True)
    concurrency = marker.kwargs.get("concurrency", 4)

    if not v1 or not v2:
        return

    from promptdiff.sdk import compare as sdk_compare

    item._promptdiff_report = sdk_compare(
        v1=v1,
        v2=v2,
        dataset=dataset,
        model=model,
        model_v1=model_v1,
        model_v2=model_v2,
        eval_metrics=eval_metrics,
        assertions=assert_rules,
        mock=mock,
        concurrency=concurrency,
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: Any) -> Any:
    """Enforce verdict pass/fail assertion after test function execution."""
    outcome = yield
    if outcome.excinfo is not None:
        return
    if hasattr(item, "_promptdiff_report"):
        rep = item._promptdiff_report
        if not rep.verdict.passed:
            reasons = rep.verdict.failed_assertions or ["Prompt regression detected"]
            raise AssertionError("PromptDiff regression detected:\n" + "\n".join(f"- {r}" for r in reasons))
