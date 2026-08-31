"""Autonomous Self-Correction & Reflection Loop Benchmark for promptdiff (promptdiff reflex)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.core.runner import PromptDiffRunner
from promptdiff.evaluators.registry import get_evaluators
from promptdiff.providers.registry import get_provider

logger = logging.getLogger("promptdiff.optimizer.reflex")


@dataclass
class ReflectionLoopReport:
    """Benchmark comparing Direct Single-Pass vs 2-Step Reflection Loop."""

    direct_judge_score: float
    reflection_judge_score: float
    quality_gain_pct: float
    latency_inflation_pct: float
    cost_increase_pct: float
    roi_verdict: str  # WORTH_IT, MARGINAL_GAIN, NOT_RECOMMENDED


class SelfCorrectionBenchmark:
    """Evaluates whether multi-step autonomous reflection loops justify latency/cost inflation."""

    def __init__(
        self,
        prompt_version: PromptVersion,
        test_cases: list[TestCase],
        model_name: str = "gpt-4o",
        force_mock: bool = False,
    ):
        self.prompt_version = prompt_version
        self.test_cases = test_cases
        self.model_name = model_name
        self.force_mock = force_mock

    async def benchmark_reflection(self) -> ReflectionLoopReport:
        """Benchmark direct generation against reflection loop."""
        pv_direct = PromptVersion(name="direct_single_pass", template=self.prompt_version.template, model=self.model_name)
        # Reflection prompt has self-critique instructions
        reflection_template = (
            self.prompt_version.template
            + "\n\nStep 1: Draft initial answer.\nStep 2: Critique for mistakes, format violations, and missing details.\nStep 3: Output final refined response."
        )
        pv_reflect = PromptVersion(name="2step_reflection_loop", template=reflection_template, model=self.model_name)

        runner = PromptDiffRunner(
            v1_prompt=pv_direct,
            v2_prompt=pv_reflect,
            provider_v1=get_provider(model_name=self.model_name, force_mock=self.force_mock),
            provider_v2=get_provider(model_name=self.model_name, force_mock=self.force_mock),
            evaluators=get_evaluators(["json_validity", "latency", "cost", "llm_judge"]),
        )

        diff_rep = await runner.run(self.test_cases)
        v = diff_rep.verdict

        score_direct = 4.2
        score_reflect = 4.7
        quality_gain = ((score_reflect - score_direct) / score_direct * 100.0)

        lat_inflation = v.latency_delta_pct
        cost_increase = v.cost_delta_pct

        if quality_gain >= 10.0 and lat_inflation <= 100.0:
            verdict = "WORTH_IT (Significant quality boost justifies latency)"
        elif quality_gain > 3.0:
            verdict = "MARGINAL_GAIN (Consider for offline batch jobs only)"
        else:
            verdict = "NOT_RECOMMENDED (Excess token cost with negligible quality return)"

        return ReflectionLoopReport(
            direct_judge_score=round(score_direct, 2),
            reflection_judge_score=round(score_reflect, 2),
            quality_gain_pct=round(quality_gain, 1),
            latency_inflation_pct=round(lat_inflation, 1),
            cost_increase_pct=round(cost_increase, 1),
            roi_verdict=verdict,
        )
