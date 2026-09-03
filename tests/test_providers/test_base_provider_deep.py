"""Deep unit tests for BaseLLMProvider and resilience helpers in providers/base.py."""

from __future__ import annotations

import httpx
import pytest

from promptdiff.providers.base import (
    BaseLLMProvider,
    ProviderResponse,
    execute_with_resilience,
    is_retryable_exception,
)


class DummyConcreteProvider(BaseLLMProvider):
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = 2048,
    ) -> ProviderResponse:
        return ProviderResponse(
            output="Hello streaming world",
            prompt_tokens=3,
            completion_tokens=3,
            total_tokens=6,
            latency_ms=15.0,
            model=self.model_name,
        )


def test_is_retryable_exception() -> None:
    # 429 status error
    req = httpx.Request("POST", "https://api.test.com")
    resp_429 = httpx.Response(429, request=req)
    err_429 = httpx.HTTPStatusError("Rate limited", request=req, response=resp_429)
    assert is_retryable_exception(err_429) is True

    # 503 status error
    resp_503 = httpx.Response(503, request=req)
    err_503 = httpx.HTTPStatusError("Service unavailable", request=req, response=resp_503)
    assert is_retryable_exception(err_503) is True

    # 400 Bad Request (not retryable)
    resp_400 = httpx.Response(400, request=req)
    err_400 = httpx.HTTPStatusError("Bad request", request=req, response=resp_400)
    assert is_retryable_exception(err_400) is False

    # Timeout
    timeout_err = httpx.ReadTimeout("Timeout", request=req)
    assert is_retryable_exception(timeout_err) is True

    # String description matching
    custom_err = RuntimeError("429 Too Many Requests: Rate limit exceeded")
    assert is_retryable_exception(custom_err) is True

    # Standard non-retryable error
    assert is_retryable_exception(ValueError("Invalid syntax")) is False


@pytest.mark.asyncio
async def test_base_provider_generate_stream() -> None:
    provider = DummyConcreteProvider(model_name="dummy-v1")
    chunks = []
    async for chunk in provider.generate_stream("Test prompt"):
        chunks.append(chunk)

    joined = "".join(chunks)
    assert joined == "Hello streaming world"


@pytest.mark.asyncio
async def test_execute_with_resilience_success() -> None:
    call_count = 0

    async def flaky_call() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RuntimeError("429 rate limit exceeded")
        return "success"

    res = await execute_with_resilience(flaky_call, max_attempts=3, min_wait=0.01, max_wait=0.05)
    assert res == "success"
    assert call_count == 2


@pytest.mark.asyncio
async def test_execute_with_resilience_unrecoverable() -> None:
    async def failing_call() -> str:
        raise ValueError("Fatal syntax error")

    with pytest.raises(ValueError, match="Fatal syntax error"):
        await execute_with_resilience(failing_call, max_attempts=3, min_wait=0.01)
