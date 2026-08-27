"""Unit tests for Providers subsystem."""

import pytest
from promptdiff.providers.mock_provider import MockProvider
from promptdiff.providers.registry import get_provider


@pytest.mark.asyncio
async def test_mock_provider_generation():
    provider = MockProvider(model_name="mock-gpt-4o", simulate_delay=False)

    # Text scenario
    resp1 = await provider.generate("You are a helpful assistant. Help customer with password reset.")
    assert len(resp1.output) > 0
    assert resp1.prompt_tokens > 0
    assert resp1.completion_tokens > 0
    assert resp1.latency_ms > 0

    # JSON scenario
    resp2 = await provider.generate("Extract customer ticket and output as JSON.")
    assert "{" in resp2.output and "}" in resp2.output


def test_provider_registry():
    # Force mock
    p_mock = get_provider("gpt-4o", force_mock=True)
    assert isinstance(p_mock, MockProvider)

    # Prefix mock
    p_mock2 = get_provider("mock-claude-3-5")
    assert isinstance(p_mock2, MockProvider)
