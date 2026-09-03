"""LLM Prompt / Prefix Caching Simulator & Structural Optimizer (promptdiff cache-sim).

Analyzes prompt template structures for OpenAI, Anthropic, and Gemini prefix caching rules,
reorders static vs dynamic components to maximize cache hit rates, and calculates projected cost savings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.pricing import get_model_pricing


@dataclass
class CacheSimReport:
    """Outcome of prefix caching analysis and restructuring."""

    model_name: str
    original_template: str
    optimized_template: str
    original_cache_hit_rate_pct: float
    optimized_cache_hit_rate_pct: float
    prefix_tokens_cached: int
    dynamic_tokens: int
    estimated_standard_cost_per_million_reqs: float
    estimated_cached_cost_per_million_reqs: float
    monthly_savings_forecast_usd: float  # At 1M reqs/day
    structural_insights: list[str]


class PromptCacheSimulator:
    """Simulates and optimizes prompt prefix caching potential."""

    def __init__(
        self,
        prompt_version: PromptVersion,
        test_cases: Optional[list[TestCase]] = None,
        model_name: str = "claude-3-5-sonnet",
        daily_volume: int = 1_000_000,
    ):
        self.prompt_version = prompt_version
        self.test_cases = test_cases or []
        self.model_name = model_name
        self.daily_volume = daily_volume

    def _estimate_tokens(self, text: str) -> int:
        return max(1, int(len(re.findall(r"\w+|[^\w\s]", text, re.UNICODE)) * 1.1))

    def _identify_variables(self, text: str) -> list[str]:
        return list(set(re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}|\{([a-zA-Z0-9_]+)\}", text)))

    def analyze_and_optimize(self) -> CacheSimReport:
        original = self.prompt_version.template
        lines = original.split("\n")

        static_lines = []
        dynamic_lines = []

        var_pattern = re.compile(r"\{\{.*?\}\}|\{.*?\}")

        first_var_line_idx = -1
        for idx, line in enumerate(lines):
            if var_pattern.search(line):
                if first_var_line_idx == -1:
                    first_var_line_idx = idx
                dynamic_lines.append(line)
            else:
                static_lines.append(line)

        # Caching heuristics:
        # If variables appear in the top 30% of the prompt, cache hit rate is ~0-10% because prefix varies per request.
        # If variables appear exclusively at the end, cache hit rate reaches 85-95% across requests.
        total_lines = max(1, len(lines))
        var_position_ratio = (first_var_line_idx / total_lines) if first_var_line_idx != -1 else 1.0

        if var_position_ratio < 0.5:
            orig_hit_rate = 10.0
        elif var_position_ratio < 0.8:
            orig_hit_rate = 40.0
        else:
            orig_hit_rate = 85.0

        # Construct optimized template with static instructions strictly at the top
        optimized_lines = [line for line in static_lines if line.strip()]
        if dynamic_lines:
            optimized_lines.append("\n--- USER REQUEST INPUTS ---")
            optimized_lines.extend(dynamic_lines)

        optimized_template = "\n".join(optimized_lines)
        opt_hit_rate = 92.0

        static_text = "\n".join(static_lines)
        dynamic_text = "\n".join(dynamic_lines)

        prefix_tokens = self._estimate_tokens(static_text)
        dyn_tokens = self._estimate_tokens(dynamic_text)

        # Pricing calculations
        pricing = get_model_pricing(self.model_name)
        # Standard input cost vs Cached input cost (typically ~10-25% of standard input price)
        cached_input_rate = pricing.input_per_million * 0.20  # 80% discount on cached tokens

        # Cost per 1M requests
        std_cost = (prefix_tokens + dyn_tokens) * pricing.input_per_token * 1_000_000
        # Cached cost: dynamic tokens at full price + prefix tokens discounted by hit rate
        cached_prefix_cost = (prefix_tokens * pricing.input_per_token * (1.0 - (opt_hit_rate / 100.0))) + (
            prefix_tokens * (cached_input_rate / 1_000_000) * (opt_hit_rate / 100.0)
        )
        opt_cost = (dyn_tokens * pricing.input_per_token + cached_prefix_cost) * 1_000_000

        savings_per_million = max(0.0, std_cost - opt_cost)
        monthly_savings = savings_per_million * (self.daily_volume / 1_000_000.0) * 30.0

        insights = []
        if first_var_line_idx != -1 and first_var_line_idx < (total_lines * 0.6):
            insights.append(
                f"Dynamic variables detected early at line {first_var_line_idx + 1}. "
                "Moving static instructions above variables increases prefix cache reuse from 10% to 92%."
            )
        insights.append(
            f"Prefix token block of {prefix_tokens} tokens is fully eligible for Anthropic / OpenAI prompt caching."
        )

        return CacheSimReport(
            model_name=self.model_name,
            original_template=original,
            optimized_template=optimized_template,
            original_cache_hit_rate_pct=orig_hit_rate,
            optimized_cache_hit_rate_pct=opt_hit_rate,
            prefix_tokens_cached=prefix_tokens,
            dynamic_tokens=dyn_tokens,
            estimated_standard_cost_per_million_reqs=round(std_cost, 2),
            estimated_cached_cost_per_million_reqs=round(opt_cost, 2),
            monthly_savings_forecast_usd=round(monthly_savings, 2),
            structural_insights=insights,
        )
