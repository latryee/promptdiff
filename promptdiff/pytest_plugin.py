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

        return asyncio.run(runner.run(test_cases))

    return _eval
