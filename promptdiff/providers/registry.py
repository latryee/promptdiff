"""Provider Registry & Factory Resolution."""

from __future__ import annotations

import os
from typing import Any

from promptdiff.providers.anthropic_provider import AnthropicProvider
from promptdiff.providers.base import BaseLLMProvider
from promptdiff.providers.gemini_provider import GeminiProvider
from promptdiff.providers.mock_provider import MockProvider
from promptdiff.providers.ollama_provider import OllamaProvider
from promptdiff.providers.openai_provider import OpenAIProvider

CUSTOM_PROVIDER_REGISTRY: dict[str, type[BaseLLMProvider]] = {}


def register_provider(name_or_prefix: str, provider_cls: type[BaseLLMProvider]) -> None:
    """Register a custom LLM provider class under a name or prefix key."""
    clean = name_or_prefix.strip().lower()
    CUSTOM_PROVIDER_REGISTRY[clean] = provider_cls


def get_provider(
    model_name: str = "gpt-4o",
    force_mock: bool = False,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> BaseLLMProvider:
    """Resolve and return appropriate BaseLLMProvider instance.

    If force_mock is True, or if model_name starts with 'mock', returns MockProvider.
    Otherwise resolves custom registered providers, OpenAI, Anthropic, Gemini, or Ollama automatically.
    """
    clean_name = model_name.strip().lower()

    if force_mock or clean_name.startswith("mock"):
        return MockProvider(model_name=model_name, **kwargs)

    # Check custom provider registry
    if clean_name in CUSTOM_PROVIDER_REGISTRY:
        return CUSTOM_PROVIDER_REGISTRY[clean_name](model_name=model_name, api_key=api_key, base_url=base_url, **kwargs)
    for prefix, prov_cls in CUSTOM_PROVIDER_REGISTRY.items():
        if clean_name.startswith(prefix):
            return prov_cls(model_name=model_name, api_key=api_key, base_url=base_url, **kwargs)

    # Gemini
    if clean_name.startswith("gemini") or "google" in clean_name:
        if not api_key and not os.getenv("GEMINI_API_KEY"):
            # If no key is set and user ran a model, graceful fallback hint or mock
            pass
        return GeminiProvider(model_name=model_name, api_key=api_key, **kwargs)

    # Claude / Anthropic
    if clean_name.startswith("claude") or "anthropic" in clean_name:
        return AnthropicProvider(model_name=model_name, api_key=api_key, **kwargs)

    # Ollama
    if clean_name.startswith("ollama/") or clean_name in ["llama3", "mistral", "qwen", "phi3"]:
        real_model = clean_name.replace("ollama/", "")
        return OllamaProvider(model_name=real_model, **kwargs)

    # OpenAI / OpenRouter / DeepSeek / Default
    return OpenAIProvider(model_name=model_name, api_key=api_key, base_url=base_url, **kwargs)
