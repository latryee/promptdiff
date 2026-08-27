"""Abstract Base Provider for LLM Integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


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
