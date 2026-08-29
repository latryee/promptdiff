"""OpenAI & OpenAI-Compatible API Provider with Resilience and Exponential Backoff.

Supports OpenAI (GPT-4o, o3-mini, GPT-4.5), OpenRouter, DeepSeek, Groq, Together, and local vLLM.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from promptdiff.providers.base import BaseLLMProvider, ProviderResponse, execute_with_resilience


class OpenAIProvider(BaseLLMProvider):
    """OpenAI / OpenAI-compatible Chat Completions Provider using async HTTPX with tenacity resilience."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout

    async def _call_api(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        """Direct HTTP post to Chat Completions endpoint."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
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
                "OPENAI_API_KEY environment variable is not set. "
                "Provide OPENAI_API_KEY or use --mock for zero-key evaluation."
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        start_time = time.perf_counter()
        data = await execute_with_resilience(
            self._call_api,
            payload=payload,
            headers=headers,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        choice = data.get("choices", [{}])[0]
        output = choice.get("message", {}).get("content", "")

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", max(1, len(prompt) // 4))
        completion_tokens = usage.get("completion_tokens", max(1, len(output) // 4))
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        return ProviderResponse(
            output=output,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 2),
            model=self.model_name,
            raw_response=data,
        )
