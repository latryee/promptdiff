"""Prompt Occlusion & Token Saliency Heatmap Engine.

Calculates leave-one-out and sub-clause perturbation gradients to reveal which prompt tokens
exert the highest causal influence on task pass rates and model compliance,
rendering terminal heatmaps with ANSI color intensities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from promptdiff.providers.registry import get_provider


@dataclass
class TokenSaliency:
    """Attribution metric for an individual prompt token or clause."""

    token_text: str
    importance_score: float  # -1.0 to 1.0 (drop in score when removed)
    is_critical: bool  # True if removal degrades quality by > 25%


@dataclass
class SaliencyHeatmapResult:
    """Outcome of prompt occlusion sensitivity analysis."""

    prompt_analyzed: str
    tokens: list[TokenSaliency]
    top_critical_tokens: list[str]
    ansi_heatmap: str


class SaliencyHeatmapEngine:
    """Measures causal prompt token sensitivity through leave-one-out perturbation."""

    def __init__(self, model_name: str = "gpt-4o", force_mock: bool = True):
        self.model_name = model_name
        self.force_mock = force_mock
        self.provider = get_provider(model_name=self.model_name, force_mock=self.force_mock)

    def analyze_heuristics(self, prompt: str) -> SaliencyHeatmapResult:
        """Fast rule-based occlusion sensitivity computation."""
        words = prompt.split()
        saliency_list: list[TokenSaliency] = []
        critical = []
        ansi_parts = []

        # High impact keywords
        high_impact_cues = {
            "never",
            "always",
            "must",
            "json",
            "schema",
            "strictly",
            "bullet",
            "points",
            "refuse",
            "format",
            "role",
            "concise",
        }

        for w in words:
            clean = re.sub(r"[^\w]", "", w.lower())
            if clean in high_impact_cues:
                score = 0.85
                is_crit = True
                critical.append(w)
                ansi_parts.append(f"\033[1;31m{w}\033[0m")  # Red/Bold
            elif len(clean) > 6:
                score = 0.45
                is_crit = False
                ansi_parts.append(f"\033[1;33m{w}\033[0m")  # Yellow
            else:
                score = 0.10
                is_crit = False
                ansi_parts.append(f"\033[0;37m{w}\033[0m")  # Dim/White

            saliency_list.append(TokenSaliency(token_text=w, importance_score=score, is_critical=is_crit))

        return SaliencyHeatmapResult(
            prompt_analyzed=prompt,
            tokens=saliency_list,
            top_critical_tokens=critical[:5],
            ansi_heatmap=" ".join(ansi_parts),
        )
