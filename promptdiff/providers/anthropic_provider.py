"""Anthropic Claude API Provider with Async Resilience.

Supports Claude 3.5 Sonnet, Claude 3.5 Haiku, and Claude 3 Opus via Messages API.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from promptdiff.providers.base import BaseLLMProvider, ProviderResponse, with_retry


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude Messages API Provider with exponential retry handling."""

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-latest",
        api_key: str | None = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.timeout = timeout

    @with_retry()
    async def _call_api(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        """Direct HTTP post to Anthropic Messages endpoint."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = 2048,
    ) -> ProviderResponse:
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Provide ANTHROPIC_API_KEY or use --mock for zero-key evaluation."
            )

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or 2048,
            "temperature": temperature,
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        start_time = time.perf_counter()
        data = await self._call_api(
            payload=payload,
            headers=headers,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        content_blocks = data.get("content", [])
        output = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")

        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", max(1, len(prompt) // 4))
        completion_tokens = usage.get("output_tokens", max(1, len(output) // 4))
        total_tokens = prompt_tokens + completion_tokens

        return ProviderResponse(
            output=output,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 2),
            model=self.model_name,
            raw_response=data,
        )
