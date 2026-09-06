"""Centralized Model Pricing Registry and Cost Forecasting Engine.

Provides cost per 1M tokens (USD) for OpenAI, Anthropic, Google Gemini, DeepSeek,
Meta Llama, Mistral, and local/free providers, plus production scale cost projection.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Union

logger = logging.getLogger("promptdiff.pricing")


PRICING_TABLE_VERSION: str = "2025.03"


@dataclass(frozen=True)
class ModelPrice:
    """Pricing structure per 1M tokens in USD."""

    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None = None
    reasoning_per_million: float | None = None
    description: str = ""

    @property
    def input_per_token(self) -> float:
        return self.input_per_million / 1_000_000.0

    @property
    def output_per_token(self) -> float:
        return self.output_per_million / 1_000_000.0

    @property
    def cached_input_per_token(self) -> float:
        if self.cached_input_per_million is not None:
            return self.cached_input_per_million / 1_000_000.0
        return self.input_per_token

    @property
    def reasoning_per_token(self) -> float:
        if self.reasoning_per_million is not None:
            return self.reasoning_per_million / 1_000_000.0
        return self.output_per_token


@dataclass(frozen=True)
class CostCalculationResult:
    """Detailed financial provenance and token calculation breakdown."""

    total_cost: float
    input_cost: float
    output_cost: float
    cached_input_cost: float
    reasoning_cost: float
    status: str  # "known", "estimated", "unavailable"
    pricing_version: str
    model: str
    explanation: str


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
    "gpt-4.5-preview": ModelPrice(75.00, 150.00, description="OpenAI GPT-4.5 preview"),
    "gpt-4.5": ModelPrice(75.00, 150.00, description="OpenAI GPT-4.5"),
    "gpt-4o": ModelPrice(2.50, 10.00, cached_input_per_million=1.25, description="OpenAI GPT-4o flagship multimodal"),
    "gpt-4o-2024-08-06": ModelPrice(2.50, 10.00, cached_input_per_million=1.25, description="OpenAI GPT-4o checkpoint"),
    "gpt-4o-mini": ModelPrice(
        0.15, 0.60, cached_input_per_million=0.075, description="OpenAI GPT-4o-mini fast & affordable"
    ),
    "gpt-4-turbo": ModelPrice(10.00, 30.00, description="OpenAI GPT-4 Turbo"),
    "gpt-4": ModelPrice(30.00, 60.00, description="OpenAI GPT-4 legacy"),
    "gpt-3.5-turbo": ModelPrice(0.50, 1.50, description="OpenAI GPT-3.5 Turbo"),
    "o1": ModelPrice(
        15.00,
        60.00,
        cached_input_per_million=7.50,
        reasoning_per_million=60.00,
        description="OpenAI o1 reasoning flagship",
    ),
    "o1-preview": ModelPrice(
        15.00,
        60.00,
        cached_input_per_million=7.50,
        reasoning_per_million=60.00,
        description="OpenAI o1 reasoning preview",
    ),
    "o1-mini": ModelPrice(
        3.00,
        12.00,
        cached_input_per_million=1.50,
        reasoning_per_million=12.00,
        description="OpenAI o1-mini fast reasoning",
    ),
    "o3-mini": ModelPrice(
        1.10,
        4.40,
        cached_input_per_million=0.55,
        reasoning_per_million=4.40,
        description="OpenAI o3-mini efficient reasoning",
    ),
    # Anthropic Claude Models
    "claude-3-7-sonnet-latest": ModelPrice(
        3.00, 15.00, cached_input_per_million=0.30, description="Anthropic Claude 3.7 Sonnet hybrid reasoning"
    ),
    "claude-3-7-sonnet": ModelPrice(
        3.00, 15.00, cached_input_per_million=0.30, description="Anthropic Claude 3.7 Sonnet"
    ),
    "claude-3-5-sonnet-20241022": ModelPrice(
        3.00, 15.00, cached_input_per_million=0.30, description="Anthropic Claude 3.5 Sonnet v2"
    ),
    "claude-3-5-sonnet-latest": ModelPrice(
        3.00, 15.00, cached_input_per_million=0.30, description="Anthropic Claude 3.5 Sonnet"
    ),
    "claude-3-5-haiku-latest": ModelPrice(
        0.80, 4.00, cached_input_per_million=0.08, description="Anthropic Claude 3.5 Haiku"
    ),
    "claude-3-5-haiku-20241022": ModelPrice(
        0.80, 4.00, cached_input_per_million=0.08, description="Anthropic Claude 3.5 Haiku"
    ),
    "claude-3-opus-latest": ModelPrice(
        15.00, 75.00, cached_input_per_million=1.50, description="Anthropic Claude 3 Opus"
    ),
    "claude-3-haiku-20240307": ModelPrice(0.25, 1.25, description="Anthropic Claude 3 Haiku legacy"),
    # Google Gemini Models
    "gemini-2.5-pro": ModelPrice(
        1.25, 5.00, cached_input_per_million=0.3125, description="Google Gemini 2.5 Pro advanced thinking"
    ),
    "gemini-2.0-flash": ModelPrice(
        0.10, 0.40, cached_input_per_million=0.025, description="Google Gemini 2.0 Flash next-gen"
    ),
    "gemini-2.0-flash-exp": ModelPrice(0.00, 0.00, description="Google Gemini 2.0 Flash Exp (Free)"),
    "gemini-1.5-pro": ModelPrice(1.25, 5.00, cached_input_per_million=0.3125, description="Google Gemini 1.5 Pro"),
    "gemini-1.5-pro-latest": ModelPrice(
        1.25, 5.00, cached_input_per_million=0.3125, description="Google Gemini 1.5 Pro Latest"
    ),
    "gemini-1.5-flash": ModelPrice(
        0.075, 0.30, cached_input_per_million=0.01875, description="Google Gemini 1.5 Flash"
    ),
    "gemini-1.5-flash-latest": ModelPrice(
        0.075, 0.30, cached_input_per_million=0.01875, description="Google Gemini 1.5 Flash Latest"
    ),
    # DeepSeek Models
    "deepseek-chat": ModelPrice(0.14, 0.28, cached_input_per_million=0.014, description="DeepSeek-V3"),
    "deepseek-v3": ModelPrice(0.14, 0.28, cached_input_per_million=0.014, description="DeepSeek-V3"),
    "deepseek-reasoner": ModelPrice(
        0.55, 2.19, cached_input_per_million=0.14, reasoning_per_million=2.19, description="DeepSeek-R1 reasoning"
    ),
    "deepseek-r1": ModelPrice(
        0.55, 2.19, cached_input_per_million=0.14, reasoning_per_million=2.19, description="DeepSeek-R1 reasoning"
    ),
    # Meta Llama (via Together/Groq/OpenRouter pricing benchmark)
    "llama-3.3-70b": ModelPrice(0.59, 0.79, description="Meta Llama 3.3 70B Instruct"),
    "llama-3.1-70b": ModelPrice(0.59, 0.79, description="Meta Llama 3.1 70B Instruct"),
    "llama-3.1-8b": ModelPrice(0.05, 0.08, description="Meta Llama 3.1 8B Instruct"),
    "llama-3.1-405b": ModelPrice(2.50, 3.50, description="Meta Llama 3.1 405B Instruct"),
    # Mistral AI
    "mistral-large-latest": ModelPrice(2.00, 6.00, description="Mistral Large 2"),
    "mistral-small-latest": ModelPrice(0.20, 0.60, description="Mistral Small"),
    "codestral-latest": ModelPrice(0.30, 0.90, description="Mistral Codestral"),
    # Local / Mock / Free
    "mock": ModelPrice(0.50, 1.50, description="Deterministic Mock Model (Simulated Pricing)"),
    "ollama": ModelPrice(0.00, 0.00, description="Local Ollama (Self-Hosted / Free)"),
    "local": ModelPrice(0.00, 0.00, description="Local Self-Hosted LLM (Free)"),
}

DEFAULT_PRICE = ModelPrice(1.00, 3.00, description="Generic default pricing")


def normalize_model_name(model_name: str) -> str:
    """Normalize model identifier strings for case-insensitive lookup."""
    return model_name.strip().lower()


def is_known_model(model_name: str) -> bool:
    """Return True if model has explicit or vendor-matched pricing in the registry."""
    clean_name = normalize_model_name(model_name)
    if clean_name in MODEL_PRICING_TABLE:
        return True
    if "/" in clean_name and clean_name.split("/")[-1] in MODEL_PRICING_TABLE:
        return True
    if any(k in clean_name or clean_name in k for k in MODEL_PRICING_TABLE):
        return True
    return False


def register_model_pricing(model_name: str, price: ModelPrice) -> None:
    """Register or override pricing for a specific model."""
    clean_name = normalize_model_name(model_name)
    MODEL_PRICING_TABLE[clean_name] = price
    logger.debug("Registered custom pricing for model '%s': %s", clean_name, price)


def load_pricing_overrides(overrides: dict[str, Any]) -> int:
    """Load and apply multiple model pricing overrides from dictionary."""
    count = 0
    for name, spec in overrides.items():
        if isinstance(spec, ModelPrice):
            register_model_pricing(name, spec)
            count += 1
        elif isinstance(spec, dict):
            price = ModelPrice(
                input_per_million=float(spec.get("input_per_million", 1.0)),
                output_per_million=float(spec.get("output_per_million", 3.0)),
                cached_input_per_million=(
                    float(spec["cached_input_per_million"]) if "cached_input_per_million" in spec else None
                ),
                reasoning_per_million=(
                    float(spec["reasoning_per_million"]) if "reasoning_per_million" in spec else None
                ),
                description=str(spec.get("description", "Custom override pricing")),
            )
            register_model_pricing(name, price)
            count += 1
    return count


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


def calculate_detailed_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> CostCalculationResult:
    """Calculate exact cost breakdown with status provenance (known, estimated, unavailable)."""
    clean_name = normalize_model_name(model_name)
    known = is_known_model(clean_name)
    pricing = get_model_pricing(clean_name)

    # Determine status
    if (
        prompt_tokens == 0
        and completion_tokens == 0
        and not clean_name.startswith("mock")
        and "ollama" not in clean_name
    ):
        status = "unavailable"
        explanation = "Token usage unavailable; reported cost is 0.0"
    elif not known and "ollama" not in clean_name and "local" not in clean_name:
        status = "estimated"
        explanation = f"Model '{model_name}' not in registry; using default estimate ($1.00 / $3.00 per 1M tokens)"
    else:
        status = "known"
        explanation = f"Exact calculation using pricing v{PRICING_TABLE_VERSION} for {model_name}"

    cached = min(cached_tokens, prompt_tokens)
    regular_input = max(0, prompt_tokens - cached)
    cached_input_cost = cached * pricing.cached_input_per_token
    input_cost = regular_input * pricing.input_per_token

    reasoning = min(reasoning_tokens, completion_tokens)
    regular_output = max(0, completion_tokens - reasoning)
    reasoning_cost = reasoning * pricing.reasoning_per_token
    output_cost = regular_output * pricing.output_per_token

    total = input_cost + cached_input_cost + output_cost + reasoning_cost
    return CostCalculationResult(
        total_cost=round(total, 6),
        input_cost=round(input_cost, 6),
        output_cost=round(output_cost, 6),
        cached_input_cost=round(cached_input_cost, 6),
        reasoning_cost=round(reasoning_cost, 6),
        status=status,
        pricing_version=PRICING_TABLE_VERSION,
        model=model_name,
        explanation=explanation,
    )


def calculate_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> float:
    """Calculate total cost in USD given model and token counts (backward-compatible)."""
    result = calculate_detailed_cost(
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )
    return result.total_cost


_TIKTOKEN_WARNED = False


def estimate_tokens(text: str, model_name: str = "gpt-4o") -> int:
    """Estimate token count for a given text.

    Uses `tiktoken` when installed (with appropriate model encoding), otherwise
    logs a warning and falls back to a regex word-count heuristic.
    """
    global _TIKTOKEN_WARNED
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except (KeyError, ValueError):
            encoding = tiktoken.get_encoding("cl100k_base")
        return max(1, len(encoding.encode(text)))
    except Exception:
        if not _TIKTOKEN_WARNED:
            logger.warning(
                "tiktoken is not installed or encoding is unavailable; falling back to regex-based token estimation. "
                "Install with `pip install 'promptdiff-eval[tokenizer]'` for exact token counts."
            )
            _TIKTOKEN_WARNED = True
        words = len(re.findall(r"\w+|[^\w\s]", text, re.UNICODE))
        return max(1, int(words * 1.1))


def calculate_text_cost(
    model_name: str,
    prompt_text: str,
    completion_text: str,
) -> float:
    """Calculate cost in USD directly from prompt and completion text using estimated tokens."""
    prompt_tokens = estimate_tokens(prompt_text, model_name=model_name)
    completion_tokens = estimate_tokens(completion_text, model_name=model_name)
    return calculate_cost(model_name, prompt_tokens, completion_tokens)


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

    delta_pct = ((v2_monthly - v1_monthly) / v1_monthly * 100.0) if v1_monthly > 0 else 0.0

    if monthly_delta < 0:
        summary = (
            f"Projected Savings: ${abs(monthly_delta):,.2f}/mo "
            f"(${annual_savings:,.2f}/yr) at {vol_daily:,} reqs/day ({delta_pct:+.1f}%)"
        )
    elif monthly_delta > 0:
        summary = f"Projected Cost Increase: +${monthly_delta:,.2f}/mo at {vol_daily:,} reqs/day ({delta_pct:+.1f}%)"
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
