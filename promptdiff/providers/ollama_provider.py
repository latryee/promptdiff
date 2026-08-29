"""Local Ollama API Provider with Async Resilience.

Supports locally hosted models (llama3, mistral, deepseek-r1, qwen) via Ollama.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from promptdiff.providers.base import BaseLLMProvider, ProviderResponse, execute_with_resilience


class OllamaProvider(BaseLLMProvider):
    """Local Ollama REST API Provider with exponential retry handling."""

    def __init__(
        self,
        model_name: str = "llama3",
        host: str | None = None,
        timeout: float = 120.0,
        **kwargs: Any,
    ):
        super().__init__(model_name, **kwargs)
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout

    async def _call_api(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Direct HTTP post to Ollama chat endpoint."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.host}/api/chat",
                json=payload,
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
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        start_time = time.perf_counter()
        data = await execute_with_resilience(
            self._call_api,
            payload=payload,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        output = data.get("message", {}).get("content", "")
        prompt_tokens = data.get("prompt_eval_count", max(1, len(prompt) // 4))
        completion_tokens = data.get("eval_count", max(1, len(output) // 4))
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
