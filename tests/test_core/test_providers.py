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


@pytest.mark.asyncio
async def test_provider_streaming_and_profiler_integration() -> None:
    """Test async generator streaming across providers and profiler integration."""
    from promptdiff.production.streaming_profiler import AsyncStreamingProfiler

    provider = MockProvider(model_name="mock-gpt-4o", simulate_delay=False)
    chunks = []
    async for chunk in provider.generate_stream("Extract customer ticket and output as JSON."):
        chunks.append(chunk)

    assert len(chunks) > 0
    full_text = "".join(chunks)
    assert "{" in full_text and "}" in full_text

    profiler = AsyncStreamingProfiler(target_ttft_sla_ms=500.0)
    report = await profiler.profile_provider_stream(
        provider=provider,
        prompt="Explain microservices in one sentence.",
    )
    assert report.total_tokens > 0
    assert report.ttft_ms >= 0.0
    assert report.model_name == "mock-gpt-4o"


@pytest.mark.asyncio
async def test_mock_provider_deterministic_seed() -> None:
    """Test MockProvider seed parameter guarantees deterministic, reproducible output and latency."""
    p_seed1 = MockProvider(model_name="mock-gpt-4o", seed=42, simulate_delay=False)
    p_seed2 = MockProvider(model_name="mock-gpt-4o", seed=42, simulate_delay=False)
    p_diff_seed = MockProvider(model_name="mock-gpt-4o", seed=99, simulate_delay=False)

    prompt = "Generic test prompt for deterministic evaluation."

    # Identical seeds must produce strictly identical outputs, tokens, latency, and hashes
    res1 = await p_seed1.generate(prompt)
    res2 = await p_seed2.generate(prompt)

    assert res1.output == res2.output
    assert res1.latency_ms == res2.latency_ms
    assert res1.prompt_tokens == res2.prompt_tokens
    assert res1.completion_tokens == res2.completion_tokens
    assert res1.raw_response["hash"] == res2.raw_response["hash"]
    assert res1.raw_response["seed"] == 42

    # Different seeds must produce different hashes and outputs for generic prompts
    res_diff = await p_diff_seed.generate(prompt)
    assert res1.raw_response["hash"] != res_diff.raw_response["hash"]
    assert res1.output != res_diff.output
    assert res_diff.raw_response["seed"] == 99

    # Registry passes seed keyword argument
    registry_provider = get_provider("mock-gpt-4o", seed=12345)
    assert isinstance(registry_provider, MockProvider)
    assert registry_provider.seed == 12345
