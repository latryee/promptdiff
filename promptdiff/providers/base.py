"""Abstract Base Provider and Resilience Decorators for LLM Integrations."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
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
