"""Local Ollama API Provider.

Supports locally hosted models (llama3, mistral, deepseek-r1, qwen) via Ollama.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional
import httpx
from promptdiff.providers.base import BaseLLMProvider, ProviderResponse


class OllamaProvider(BaseLLMProvider):
    """Local Ollama REST API Provider."""

    def __init__(
        self,
        model_name: str = "llama3",
        host: Optional[str] = None,
        timeout: float = 120.0,
        **kwargs: Any,
    ):
        super().__init__(model_name, **kwargs)
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 2048,
    ) -> ProviderResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
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
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.host}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        output = data.get("message", {}).get("content", "")
        prompt_tokens = data.get("prompt_eval_count", len(prompt) // 4)
        completion_tokens = data.get("eval_count", len(output) // 4)
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
