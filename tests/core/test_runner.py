"""Unit & Integration tests for PromptDiffRunner."""

import pytest

from promptdiff.core.cache import DiskCache
from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.core.runner import PromptDiffRunner
from promptdiff.evaluators.registry import get_evaluators
from promptdiff.providers.mock_provider import MockProvider


@pytest.mark.asyncio
async def test_runner_execution_flow(tmp_path):
    v1 = PromptVersion(
        name="v1",
        template="You are a helpful customer support agent. Answer: {{query}}",
        model="gpt-4o",
    )
    v2 = PromptVersion(
        name="v2",
        template="You are a fast customer support agent. Answer concisely in bullets: {{query}}",
        model="gpt-4o",
    )
    p1 = MockProvider(model_name="mock-v1", simulate_delay=False)
    p2 = MockProvider(model_name="mock-v2", simulate_delay=False)

    cache = DiskCache(cache_dir=tmp_path / "cache", enabled=True)
    evaluators = get_evaluators(["json_validity", "latency", "cost", "similarity"])

    runner = PromptDiffRunner(
        v1_prompt=v1,
        v2_prompt=v2,
        provider_v1=p1,
        provider_v2=p2,
        evaluators=evaluators,
        assertions=["cost_delta <= 50%"],
        cache=cache,
        concurrency=2,
    )

    test_cases = [
        TestCase(id="tc_1", description="Inquiry 1", vars={"query": "How to reset password?"}),
        TestCase(id="tc_2", description="Inquiry 2", vars={"query": "Where is my invoice?"}),
    ]

    report = await runner.run(test_cases)

    assert report is not None
    assert report.total_cases == 2
    assert len(report.comparisons) == 2
    assert report.verdict.passed is True
    assert "cost_delta_pct" in report.aggregate_stats

    # Check that individual comparison results have diffs and scores
    c1 = report.comparisons[0]
    assert len(c1.text_diff) > 0
    assert "similarity" in c1.scores
    assert "cost" in c1.scores
    assert "latency" in c1.scores


class SlowMockProvider(MockProvider):
    async def generate(self, prompt: str, **kwargs):
        import asyncio

        from promptdiff.providers.base import ProviderResponse

        await asyncio.sleep(0.1)
        return ProviderResponse(
            output="slow", prompt_tokens=10, completion_tokens=10, total_tokens=20, latency_ms=100.0
        )


@pytest.mark.asyncio
async def test_runner_timeout(tmp_path):
    from promptdiff.core.models import ErrorCategory

    v1 = PromptVersion(name="v1", template="Test {{x}}", model="gpt-4o")
    v2 = PromptVersion(name="v2", template="Test {{x}}", model="gpt-4o")
    p1 = SlowMockProvider("slow-v1", simulate_delay=False)
    p2 = MockProvider("mock-v2", simulate_delay=False)
    cache = DiskCache(cache_dir=tmp_path / "cache", enabled=False)

    runner = PromptDiffRunner(
        v1_prompt=v1,
        v2_prompt=v2,
        provider_v1=p1,
        provider_v2=p2,
        cache=cache,
        timeout=0.03,
    )
    test_cases = [TestCase(id="tc_1", vars={"x": "1"})]
    report = await runner.run(test_cases)
    assert report.total_cases == 1
    c1 = report.comparisons[0]
    assert c1.v1_result.error is not None
    assert c1.v1_result.error_category == ErrorCategory.TIMEOUT
    assert c1.v2_result.error is None
    assert c1.v2_result.error_category == ErrorCategory.SUCCESS


@pytest.mark.asyncio
async def test_runner_provenance_and_hashes(tmp_path):
    v1 = PromptVersion(name="v1", template="Hello {{name}}", model="gpt-4o")
    v2 = PromptVersion(name="v2", template="Greetings {{name}}", model="gpt-4o")
    p1 = MockProvider("mock-v1", simulate_delay=False)
    p2 = MockProvider("mock-v2", simulate_delay=False)
    cache = DiskCache(cache_dir=tmp_path / "cache", enabled=False)

    runner = PromptDiffRunner(
        v1_prompt=v1,
        v2_prompt=v2,
        provider_v1=p1,
        provider_v2=p2,
        cache=cache,
        experiment_id="exp-123",
    )
    test_cases = [TestCase(id="tc_1", vars={"name": "Alice"})]
    report = await runner.run(test_cases)
    assert report.experiment_id == "exp-123"
    assert report.v1_prompt_hash == v1.compute_hash()
    assert report.v2_prompt_hash == v2.compute_hash()
    assert report.dataset_hash is not None and len(report.dataset_hash) == 64
    assert report.provenance is not None
    assert report.provenance.runner_version is not None
    assert report.config_snapshot is not None
