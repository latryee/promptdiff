"""Coverage boost tests for providers/base.py — ProviderResponse, retry logic, BaseLLMProvider."""

from __future__ import annotations

import httpx
import pytest

from promptdiff.providers.base import (
    ProviderResponse,
    execute_with_resilience,
    is_retryable_exception,
)


def test_provider_response_dataclass() -> None:
    resp = ProviderResponse(
        output="Hello world",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        latency_ms=42.0,
        model="gpt-4o",
    )
    assert resp.output == "Hello world"
    assert resp.total_tokens == 15
    assert resp.latency_ms == 42.0


def test_is_retryable_rate_limit() -> None:
    mock_resp = httpx.Response(status_code=429, request=httpx.Request("POST", "http://api.test"))
    exc = httpx.HTTPStatusError("rate limited", request=mock_resp.request, response=mock_resp)
    assert is_retryable_exception(exc) is True


def test_is_retryable_server_error() -> None:
    for code in [500, 502, 503, 504]:
        mock_resp = httpx.Response(status_code=code, request=httpx.Request("POST", "http://api.test"))
        exc = httpx.HTTPStatusError("server error", request=mock_resp.request, response=mock_resp)
        assert is_retryable_exception(exc) is True


def test_is_not_retryable_client_error() -> None:
    mock_resp = httpx.Response(status_code=400, request=httpx.Request("POST", "http://api.test"))
    exc = httpx.HTTPStatusError("bad request", request=mock_resp.request, response=mock_resp)
    assert is_retryable_exception(exc) is False


def test_is_retryable_timeout() -> None:
    exc = httpx.TimeoutException("request timed out")
    assert is_retryable_exception(exc) is True


def test_is_retryable_string_match() -> None:
    exc = Exception("rate limit exceeded, please retry")
    assert is_retryable_exception(exc) is True
    exc2 = Exception("something weird")
    assert is_retryable_exception(exc2) is False


@pytest.mark.asyncio
async def test_execute_with_resilience_success() -> None:
    call_count = 0

    async def succeeding_func() -> str:
        nonlocal call_count
        call_count += 1
        return "ok"

    result = await execute_with_resilience(succeeding_func, max_attempts=3)
    assert result == "ok"
    assert call_count == 1


@pytest.mark.asyncio
async def test_execute_with_resilience_non_retryable_raises() -> None:
    async def failing_func() -> str:
        raise ValueError("non-retryable error")

    with pytest.raises(ValueError, match="non-retryable"):
        await execute_with_resilience(failing_func, max_attempts=2)


@pytest.mark.asyncio
async def test_base_provider_generate_stream() -> None:
    from promptdiff.providers.mock_provider import MockProvider

    provider = MockProvider(model_name="mock-gpt-4o")
    chunks: list[str] = []
    async for chunk in provider.generate_stream("Hello"):
        chunks.append(chunk)
    assert len(chunks) > 0
    reconstructed = "".join(chunks)
    assert len(reconstructed) > 0


@pytest.mark.asyncio
async def test_with_retry_decorator_retries_transient_failures() -> None:
    """Test @with_retry decorator retries on retryable exceptions."""
    from promptdiff.providers.base import with_retry

    attempts = 0

    @with_retry(max_attempts=3, min_wait=0.01, max_wait=0.05)
    async def flaky_api(call_id: str) -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            mock_resp = httpx.Response(status_code=429, request=httpx.Request("POST", "http://test.api"))
            raise httpx.HTTPStatusError("rate limit exceeded", request=mock_resp.request, response=mock_resp)
        return f"success-{call_id}"

    res = await flaky_api("test-123")
    assert res == "success-test-123"
    assert attempts == 3
