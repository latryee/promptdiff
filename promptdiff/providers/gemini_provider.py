"""Google Gemini API Provider.

Supports Gemini 2.0 Flash, Gemini 1.5 Pro, and Gemini 1.5 Flash.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional
import httpx
from promptdiff.providers.base import BaseLLMProvider, ProviderResponse


class GeminiProvider(BaseLLMProvider):
    """Google Gemini REST API Provider."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.timeout = timeout

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 2048,
    ) -> ProviderResponse:
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Provide GEMINI_API_KEY or use --mock for zero-key evaluation."
            )

        endpoint_model = self.model_name
        if not endpoint_model.startswith("models/"):
            endpoint_model = f"models/{endpoint_model}"

        url = f"https://generativelanguage.googleapis.com/v1beta/{endpoint_model}:generateContent?key={self.api_key}"

        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            },
        }
        if max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        candidates = data.get("candidates", [])
        output = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            output = "".join(p.get("text", "") for p in parts)

        usage_meta = data.get("usageMetadata", {})
        prompt_tokens = usage_meta.get("promptTokenCount", len(prompt) // 4)
        completion_tokens = usage_meta.get("candidatesTokenCount", len(output) // 4)
        total_tokens = usage_meta.get("totalTokenCount", prompt_tokens + completion_tokens)

        return ProviderResponse(
            output=output,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 2),
            model=self.model_name,
            raw_response=data,
        )
