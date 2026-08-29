"""Centralized Model Pricing Registry and Cost Forecasting Engine.

Provides cost per 1M tokens (USD) for OpenAI, Anthropic, Google Gemini, DeepSeek,
Meta Llama, Mistral, and local/free providers, plus production scale cost projection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union


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


@dataclass(frozen=True)
class CostForecast:
    """Production scale cost impact projection."""

    daily_volume: int
    monthly_volume: int
    annual_volume: int
    v1_avg_cost_per_req: float
    v2_avg_cost_per_req: float
    v1_monthly_cost: float
    v2_monthly_cost: float
    monthly_delta_cost: float
    monthly_savings_usd: float
    annual_savings_usd: float
    cost_delta_pct: float
    summary_text: str


# Pricing Registry (Per 1 Million Tokens in USD)
MODEL_PRICING_TABLE: dict[str, ModelPrice] = {
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

DEFAULT_PRICE = ModelPrice(1.00, 3.00, "Generic default pricing")


def normalize_model_name(model_name: str) -> str:
    """Normalize model identifier strings for case-insensitive lookup."""
    return model_name.strip().lower()


def get_model_pricing(model_name: str) -> ModelPrice:
    """Lookup model pricing from registry with fuzzy/prefix matching."""
    clean_name = normalize_model_name(model_name)

    if clean_name in MODEL_PRICING_TABLE:
        return MODEL_PRICING_TABLE[clean_name]

    if "/" in clean_name:
        vendor_stripped = clean_name.split("/")[-1]
        if vendor_stripped in MODEL_PRICING_TABLE:
            return MODEL_PRICING_TABLE[vendor_stripped]

    for key, price in MODEL_PRICING_TABLE.items():
        if key in clean_name or clean_name in key:
            return price

    if "ollama" in clean_name or "local" in clean_name:
        return MODEL_PRICING_TABLE["ollama"]

    return DEFAULT_PRICE


def calculate_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Calculate exact total cost in USD given model and token counts."""
    pricing = get_model_pricing(model_name)
    input_cost = prompt_tokens * pricing.input_per_token
    output_cost = completion_tokens * pricing.output_per_token
    total = input_cost + output_cost
    return round(total, 6)


def parse_volume_string(vol: Union[str, int, float]) -> int:
    """Parse volume strings like '1M', '500k', '2.5M', '100000' into integer count."""
    if isinstance(vol, (int, float)):
        return max(1, int(vol))

    cleaned = str(vol).strip().lower().replace(",", "")
    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([kmbt])?$", cleaned)
    if not match:
        try:
            return max(1, int(float(cleaned)))
        except ValueError:
            return 100_000

    num = float(match.group(1))
    unit = match.group(2) or ""

    multiplier = 1
    if unit == "k":
        multiplier = 1_000
    elif unit == "m":
        multiplier = 1_000_000
    elif unit == "b":
        multiplier = 1_000_000_000
    elif unit == "t":
        multiplier = 1_000_000_000_000

    return max(1, int(num * multiplier))


def calculate_forecast(
    total_cost_v1: float,
    total_cost_v2: float,
    total_cases: int,
    daily_volume: Union[str, int],
) -> CostForecast:
    """Calculate projected monthly and annual production cost impact."""
    vol_daily = parse_volume_string(daily_volume)
    cases = max(1, total_cases)

    v1_avg = total_cost_v1 / cases
    v2_avg = total_cost_v2 / cases

    monthly_vol = vol_daily * 30
    annual_vol = vol_daily * 365

    v1_monthly = v1_avg * monthly_vol
    v2_monthly = v2_avg * monthly_vol

    monthly_delta = v2_monthly - v1_monthly
    monthly_savings = -monthly_delta if monthly_delta < 0 else 0.0
    annual_savings = monthly_savings * 12.0

    delta_pct = (
        ((v2_monthly - v1_monthly) / v1_monthly * 100.0)
        if v1_monthly > 0
        else 0.0
    )

    if monthly_delta < 0:
        summary = (
            f"Projected Savings: ${abs(monthly_delta):,.2f}/mo "
            f"(${annual_savings:,.2f}/yr) at {vol_daily:,} reqs/day ({delta_pct:+.1f}%)"
        )
    elif monthly_delta > 0:
        summary = (
            f"Projected Cost Increase: +${monthly_delta:,.2f}/mo "
            f"at {vol_daily:,} reqs/day ({delta_pct:+.1f}%)"
        )
    else:
        summary = f"Zero Cost Variance at {vol_daily:,} reqs/day"

    return CostForecast(
        daily_volume=vol_daily,
        monthly_volume=monthly_vol,
        annual_volume=annual_vol,
        v1_avg_cost_per_req=round(v1_avg, 6),
        v2_avg_cost_per_req=round(v2_avg, 6),
        v1_monthly_cost=round(v1_monthly, 2),
        v2_monthly_cost=round(v2_monthly, 2),
        monthly_delta_cost=round(monthly_delta, 2),
        monthly_savings_usd=round(monthly_savings, 2),
        annual_savings_usd=round(annual_savings, 2),
        cost_delta_pct=round(delta_pct, 2),
        summary_text=summary,
    )
