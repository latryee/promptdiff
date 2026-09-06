"""Abstract Base Provider and Resilience Decorators for LLM Integrations."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Optional, TypeVar

import httpx

logger = logging.getLogger("promptdiff.providers")

# Check if tenacity is available, otherwise provide transparent fallback
try:
    from tenacity import (
        AsyncRetrying,
        retry_if_exception,
        stop_after_attempt,
        wait_random_exponential,
    )

    TENACITY_AVAILABLE = True
except ImportError:  # pragma: no cover
    TENACITY_AVAILABLE = False


@dataclass(frozen=True)
class ProviderCapabilities:
    """Exposes supported feature capabilities for an LLM provider."""

    streaming: bool = True
    system_prompt: bool = True
    temperature: bool = True
    max_tokens: bool = True
    structured_output: bool = False
    prompt_caching: bool = False
    reasoning_tokens: bool = False


@dataclass
class ProviderResponse:
    """Structured response from an LLM provider call."""

    output: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    model: str
    raw_response: Optional[Any] = None
    finish_reason: Optional[str] = None
    request_id: Optional[str] = None
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    provider_name: Optional[str] = None


def classify_provider_exception(exc: BaseException) -> Any:
    """Classify any provider exception into standard ErrorCategory."""
    from promptdiff.core.models import ErrorCategory

    if isinstance(exc, (asyncio.TimeoutError, httpx.TimeoutException)):
        return ErrorCategory.TIMEOUT
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code in {401, 403}:
            return ErrorCategory.AUTHENTICATION
        if code == 429:
            return ErrorCategory.RATE_LIMIT
        if 400 <= code < 500:
            return ErrorCategory.INVALID_REQUEST
        if code >= 500:
            return ErrorCategory.PROVIDER_ERROR
    msg = str(exc).lower()
    if any(k in msg for k in ["auth", "unauthorized", "forbidden", "api key", "invalid key", "token"]):
        return ErrorCategory.AUTHENTICATION
    if any(k in msg for k in ["rate limit", "429", "too many requests", "quota"]):
        return ErrorCategory.RATE_LIMIT
    if any(k in msg for k in ["timeout", "timed out"]):
        return ErrorCategory.TIMEOUT
    if any(k in msg for k in ["invalid", "bad request", "unknown parameter"]):
        return ErrorCategory.INVALID_REQUEST
    return ErrorCategory.PROVIDER_ERROR


def is_retryable_exception(exc: BaseException) -> bool:
    """Determine whether an exception qualifies for automatic exponential retry.

    Retries on HTTP 429 (Rate Limit), 500/502/503/504 (Server Errors), and network timeouts/disconnects.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code in {429, 500, 502, 503, 504, 520, 521, 522, 524}
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError, httpx.NetworkError, httpx.ConnectError)):
        return True
    # Catch string error descriptions from other client SDKs
    msg = str(exc).lower()
    return any(
        k in msg for k in ["rate limit", "429", "too many requests", "overloaded", "503", "500", "timeout", "timed out"]
    )


T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


def with_retry(
    max_attempts: int = 5,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
) -> Callable[[F], F]:
    """Centralized decorator to retry asynchronous provider functions with exponential backoff and jitter."""

    def decorator(func: F) -> F:
        from functools import wraps

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await execute_with_resilience(
                func,
                *args,
                max_attempts=max_attempts,
                min_wait=min_wait,
                max_wait=max_wait,
                **kwargs,
            )

        return wrapper  # type: ignore[return-value]

    return decorator


async def execute_with_resilience(
    func: Callable[..., Any],
    *args: Any,
    max_attempts: int = 5,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    **kwargs: Any,
) -> Any:
    """Execute an async provider call with exponential backoff, jitter, and automatic retry."""
    if TENACITY_AVAILABLE:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_random_exponential(multiplier=min_wait, max=max_wait),
            retry=retry_if_exception(is_retryable_exception),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                return await func(*args, **kwargs)
    else:  # Fallback basic exponential backoff
        for attempt_idx in range(1, max_attempts + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt_idx >= max_attempts or not is_retryable_exception(e):
                    raise
                backoff = min(max_wait, min_wait * (2 ** (attempt_idx - 1)))
                logger.warning(f"Retry attempt {attempt_idx}/{max_attempts} after error: {e}. Backoff {backoff:.2f}s")
                await asyncio.sleep(backoff)


class BaseLLMProvider(ABC):
    """Abstract interface that all promptdiff LLM providers must implement."""

    def __init__(self, model_name: str, **kwargs: Any):
        self.model_name = model_name
        self.kwargs = kwargs

    def get_capabilities(self) -> ProviderCapabilities:
        """Return supported capabilities for this provider."""
        return ProviderCapabilities()

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 2048,
    ) -> ProviderResponse:
        """Execute prompt against model and return normalized response."""
        raise NotImplementedError

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 2048,
    ) -> AsyncIterator[str]:
        """Stream response chunks asynchronously. Default implementation chunks generate() output."""
        resp = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        words = resp.output.split(" ")
        for i, word in enumerate(words):
            yield word if i == 0 else f" {word}"
