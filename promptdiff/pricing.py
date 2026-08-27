"""Centralized Model Pricing Registry for LLM Token Cost Calculations.

Provides cost per 1M tokens (USD) for OpenAI, Anthropic, Google Gemini, DeepSeek,
Meta Llama, Mistral, and local/free providers.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ModelPrice:
    """Pricing structure per 1M tokens in USD."""

    input_per_million: float
    output_per_million: float
    description: str = ""

    @property
    def input_per_token(self) -> float:
        return self.input_per_million / 1_000_000.0

    @property
    def output_per_token(self) -> float:
        return self.output_per_million / 1_000_000.0


# Pricing Registry (Per 1 Million Tokens in USD)
MODEL_PRICING_TABLE: Dict[str, ModelPrice] = {
    # OpenAI Models
    "gpt-4o": ModelPrice(2.50, 10.00, "OpenAI GPT-4o flagship multimodal"),
    "gpt-4o-2024-08-06": ModelPrice(2.50, 10.00, "OpenAI GPT-4o checkpoint"),
    "gpt-4o-mini": ModelPrice(0.15, 0.60, "OpenAI GPT-4o-mini fast & affordable"),
    "gpt-4-turbo": ModelPrice(10.00, 30.00, "OpenAI GPT-4 Turbo"),
    "gpt-4": ModelPrice(30.00, 60.00, "OpenAI GPT-4 legacy"),
    "gpt-3.5-turbo": ModelPrice(0.50, 1.50, "OpenAI GPT-3.5 Turbo"),
    "o1-preview": ModelPrice(15.00, 60.00, "OpenAI o1 reasoning preview"),
    "o1-mini": ModelPrice(3.00, 12.00, "OpenAI o1-mini fast reasoning"),
    "o3-mini": ModelPrice(1.10, 4.40, "OpenAI o3-mini efficient reasoning"),

    # Anthropic Claude Models
    "claude-3-5-sonnet-20241022": ModelPrice(3.00, 15.00, "Anthropic Claude 3.5 Sonnet v2"),
    "claude-3-5-sonnet-latest": ModelPrice(3.00, 15.00, "Anthropic Claude 3.5 Sonnet"),
    "claude-3-5-haiku-latest": ModelPrice(0.80, 4.00, "Anthropic Claude 3.5 Haiku"),
    "claude-3-5-haiku-20241022": ModelPrice(0.80, 4.00, "Anthropic Claude 3.5 Haiku"),
    "claude-3-opus-latest": ModelPrice(15.00, 75.00, "Anthropic Claude 3 Opus"),
    "claude-3-haiku-20240307": ModelPrice(0.25, 1.25, "Anthropic Claude 3 Haiku legacy"),

    # Google Gemini Models
    "gemini-2.0-flash": ModelPrice(0.10, 0.40, "Google Gemini 2.0 Flash next-gen"),
    "gemini-2.0-flash-exp": ModelPrice(0.00, 0.00, "Google Gemini 2.0 Flash Exp (Free)"),
    "gemini-1.5-pro": ModelPrice(1.25, 5.00, "Google Gemini 1.5 Pro"),
    "gemini-1.5-pro-latest": ModelPrice(1.25, 5.00, "Google Gemini 1.5 Pro Latest"),
    "gemini-1.5-flash": ModelPrice(0.075, 0.30, "Google Gemini 1.5 Flash"),
    "gemini-1.5-flash-latest": ModelPrice(0.075, 0.30, "Google Gemini 1.5 Flash Latest"),

    # DeepSeek Models
    "deepseek-chat": ModelPrice(0.14, 0.28, "DeepSeek-V3"),
    "deepseek-v3": ModelPrice(0.14, 0.28, "DeepSeek-V3"),
    "deepseek-reasoner": ModelPrice(0.55, 2.19, "DeepSeek-R1 reasoning"),
    "deepseek-r1": ModelPrice(0.55, 2.19, "DeepSeek-R1 reasoning"),

    # Meta Llama (via Together/Groq/OpenRouter pricing benchmark)
    "llama-3.3-70b": ModelPrice(0.59, 0.79, "Meta Llama 3.3 70B Instruct"),
    "llama-3.1-70b": ModelPrice(0.59, 0.79, "Meta Llama 3.1 70B Instruct"),
    "llama-3.1-8b": ModelPrice(0.05, 0.08, "Meta Llama 3.1 8B Instruct"),
    "llama-3.1-405b": ModelPrice(2.50, 3.50, "Meta Llama 3.1 405B Instruct"),

    # Mistral AI
    "mistral-large-latest": ModelPrice(2.00, 6.00, "Mistral Large 2"),
    "mistral-small-latest": ModelPrice(0.20, 0.60, "Mistral Small"),
    "codestral-latest": ModelPrice(0.30, 0.90, "Mistral Codestral"),

    # Local / Mock / Free
    "mock": ModelPrice(0.50, 1.50, "Deterministic Mock Model (Simulated Pricing)"),
    "ollama": ModelPrice(0.00, 0.00, "Local Ollama (Self-Hosted / Free)"),
    "local": ModelPrice(0.00, 0.00, "Local Self-Hosted LLM (Free)"),
}

# Generic Fallback Default Pricing (equivalent to standard mid-tier model)
DEFAULT_PRICE = ModelPrice(1.00, 3.00, "Generic default pricing")


def normalize_model_name(model_name: str) -> str:
    """Normalize model identifier strings for case-insensitive lookup."""
    return model_name.strip().lower()


def get_model_pricing(model_name: str) -> ModelPrice:
    """Lookup model pricing from registry with fuzzy/prefix matching.

    Args:
        model_name: Name of the model (e.g., 'gpt-4o', 'claude-3-5-sonnet', 'gemini-1.5-pro')

    Returns:
        ModelPrice instance with per-million token rates.
    """
    clean_name = normalize_model_name(model_name)

    # 1. Exact match
    if clean_name in MODEL_PRICING_TABLE:
        return MODEL_PRICING_TABLE[clean_name]

    # 2. Strip vendor prefix (e.g. 'openai/gpt-4o' -> 'gpt-4o')
    if "/" in clean_name:
        vendor_stripped = clean_name.split("/")[-1]
        if vendor_stripped in MODEL_PRICING_TABLE:
            return MODEL_PRICING_TABLE[vendor_stripped]

    # 3. Substring / Prefix matching
    for key, price in MODEL_PRICING_TABLE.items():
        if key in clean_name or clean_name in key:
            return price

    # 4. Fallback for local models
    if "ollama" in clean_name or "local" in clean_name:
        return MODEL_PRICING_TABLE["ollama"]

    return DEFAULT_PRICE


def calculate_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Calculate exact total cost in USD given model and token counts.

    Args:
        model_name: LLM model identifier.
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of output/completion tokens.

    Returns:
        Cost in USD (rounded to 6 decimal places).
    """
    pricing = get_model_pricing(model_name)
    input_cost = prompt_tokens * pricing.input_per_token
    output_cost = completion_tokens * pricing.output_per_token
    total = input_cost + output_cost
    return round(total, 6)
