"""KV-Cache & Prompt Caching Breakpoint Impact Analyzer.

Quantifies prompt prefix divergence between prompt revisions and computes financial
impact on vendor prompt caching (Anthropic Claude 1024-token boundary, OpenAI, DeepSeek).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from promptdiff.pricing import get_model_pricing


@dataclass
class CacheBreakpointResult:
    """Detailed report on prompt prefix cache preservation and financial impact."""

    model_name: str
    v1_tokens: int
    v2_tokens: int
    common_prefix_tokens: int
    common_prefix_chars: int
    breakpoint_token_idx: int
    min_cache_threshold_tokens: int
    v1_cache_eligible: bool
    v2_cache_eligible: bool
    cache_preserved: bool
    lost_cached_tokens: int
    cost_delta_per_request_usd: float
    monthly_financial_impact_usd: float
    recommendation: str


class PromptCacheImpactAnalyzer:
    """Analyzes KV-cache prefix divergence and forecasts financial cache invalidation impact."""

    VENDOR_CACHE_MIN_TOKENS: dict[str, int] = {
        "anthropic": 1024,
        "openai": 1024,
        "deepseek": 64,
        "google": 2048,
        "default": 1024,
    }

    # Standard discount multipliers for cached prompt tokens
    VENDOR_CACHE_DISCOUNT_RATIO: dict[str, float] = {
        "anthropic": 0.90,  # 90% discount on cache read
        "openai": 0.50,  # 50% discount on cached input
        "deepseek": 0.75,  # 75% discount
        "google": 0.75,  # 75% discount
        "default": 0.50,
    }

    def __init__(self, model_name: str = "claude-3-5-sonnet"):
        self.model_name = model_name
        self.vendor = self._resolve_vendor(model_name)
        self.min_cache_tokens = self.VENDOR_CACHE_MIN_TOKENS.get(self.vendor, self.VENDOR_CACHE_MIN_TOKENS["default"])
        self.discount_ratio = self.VENDOR_CACHE_DISCOUNT_RATIO.get(
            self.vendor, self.VENDOR_CACHE_DISCOUNT_RATIO["default"]
        )

    def _resolve_vendor(self, model: str) -> str:
        clean = model.lower()
        if "claude" in clean or "anthropic" in clean:
            return "anthropic"
        if "gpt" in clean or "o1" in clean or "o3" in clean or "openai" in clean:
            return "openai"
        if "deepseek" in clean:
            return "deepseek"
        if "gemini" in clean or "google" in clean:
            return "google"
        return "default"

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace and punctuation tokenizer estimating tokens accurately."""
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    def analyze(
        self,
        v1_prompt: str,
        v2_prompt: str,
        daily_requests: int = 10_000,
    ) -> CacheBreakpointResult:
        """Analyze prefix divergence between two prompt revisions."""
        tokens_v1 = self._tokenize(v1_prompt)
        tokens_v2 = self._tokenize(v2_prompt)

        len_v1 = len(tokens_v1)
        len_v2 = len(tokens_v2)

        # Find common prefix length
        common_count = 0
        min_len = min(len_v1, len_v2)
        while common_count < min_len and tokens_v1[common_count] == tokens_v2[common_count]:
            common_count += 1

        diverged_char_idx = 0
        for i in range(min(len(v1_prompt), len(v2_prompt))):
            if v1_prompt[i] == v2_prompt[i]:
                diverged_char_idx += 1
            else:
                break

        v1_eligible = len_v1 >= self.min_cache_tokens
        v2_eligible = len_v2 >= self.min_cache_tokens
        cache_preserved = common_count >= self.min_cache_tokens

        potential_cached_v1 = len_v1 if v1_eligible else 0
        # Lost cached tokens due to prefix cache invalidation
        if cache_preserved:
            lost_tokens = 0
        else:
            lost_tokens = potential_cached_v1

        # Calculate pricing delta
        pricing = get_model_pricing(self.model_name)
        input_price_per_token = pricing.input_per_token if pricing else (3.0 / 1_000_000.0)

        # Financial impact per request from lost discount
        cost_delta_per_req = lost_tokens * input_price_per_token * self.discount_ratio
        monthly_impact = cost_delta_per_req * daily_requests * 30.0

        recommendation = self._build_recommendation(
            common_count=common_count,
            len_v1=len_v1,
            len_v2=len_v2,
            v1_eligible=v1_eligible,
            v2_eligible=v2_eligible,
            cache_preserved=cache_preserved,
        )

        return CacheBreakpointResult(
            model_name=self.model_name,
            v1_tokens=len_v1,
            v2_tokens=len_v2,
            common_prefix_tokens=common_count,
            common_prefix_chars=diverged_char_idx,
            breakpoint_token_idx=common_count,
            min_cache_threshold_tokens=self.min_cache_tokens,
            v1_cache_eligible=v1_eligible,
            v2_cache_eligible=v2_eligible,
            cache_preserved=cache_preserved,
            lost_cached_tokens=lost_tokens,
            cost_delta_per_request_usd=round(cost_delta_per_req, 6),
            monthly_financial_impact_usd=round(monthly_impact, 2),
            recommendation=recommendation,
        )

    def _build_recommendation(
        self,
        common_count: int,
        len_v1: int,
        len_v2: int,
        v1_eligible: bool,
        v2_eligible: bool,
        cache_preserved: bool,
    ) -> str:
        if not v1_eligible and not v2_eligible:
            return (
                f"Prompts are below the {self.vendor.capitalize()} minimum cacheable prefix threshold "
                f"({self.min_cache_tokens} tokens). Prompt caching is not active."
            )
        if cache_preserved:
            return (
                f"Optimal KV-cache preservation! The first {common_count} tokens remain identical, "
                f"exceeding the {self.min_cache_tokens}-token threshold. Cache discount maintained."
            )
        if common_count == 0:
            return (
                "Critical cache breakage: Prompt templates diverge immediately on the first token. "
                "Move static system directives and few-shot exemplars to the top before dynamic user variables."
            )
        return (
            f"Prefix cache invalidated at token index {common_count} (needs {self.min_cache_tokens}+ tokens). "
            f"Preserve existing static instructions at the header to retain vendor KV-cache discounts."
        )


def analyze_cache_impact(
    v1_prompt: str,
    v2_prompt: str,
    model: str = "claude-3-5-sonnet",
    daily_volume: int = 10_000,
) -> CacheBreakpointResult:
    """Analyze KV-cache prefix divergence and calculate financial impact between prompt revisions."""
    analyzer = PromptCacheImpactAnalyzer(model_name=model)
    return analyzer.analyze(v1_prompt, v2_prompt, daily_requests=daily_volume)
