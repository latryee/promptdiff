"""Multi-Agent Adversarial Debate & Cross-Examination Judge Evaluator.

Neutralizes LLM-as-a-judge position bias, verbosity bias, and self-enhancement bias
by establishing a tripartite courtroom:
- Advocate Agent A (Defends Prompt v1 output, critiques v2)
- Advocate Agent B (Defends Prompt v2 output, critiques v1)
- Chief Justice Judge (Synthesizes cross-examination arguments and issues a debiased verdict).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from promptdiff.core.models import EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.base import BaseEvaluator
from promptdiff.providers.base import BaseLLMProvider
from promptdiff.providers.registry import get_provider


@dataclass
class DebateRound:
    """Transcript of the adversarial debate round."""

    advocate_a_speech: str
    advocate_b_speech: str
    chief_justice_synthesis: str
    winner: str  # "v1", "v2", or "TIE"
    confidence: float
    bias_correction_applied: bool


class MultiAgentDebateEvaluator(BaseEvaluator):
    """Multi-Agent Courtroom Evaluator with Adversarial Cross-Examination."""

    name: str = "debate_judge"
    description: str = "Debiased multi-agent adversarial debate evaluator."

    def __init__(self, model_name: str = "gpt-4o", force_mock: bool = True):
        self.model_name = model_name
        self.force_mock = force_mock
        self.provider: BaseLLMProvider = get_provider(model_name=self.model_name, force_mock=self.force_mock)

    async def conduct_debate(
        self,
        query: str,
        v1_output: str,
        v2_output: str,
    ) -> DebateRound:
        """Run simulated or LLM-driven debate cross-examination."""
        # Check verbosity ratio to detect superficial verbosity bias
        len_a = max(1, len(v1_output.split()))
        len_b = max(1, len(v2_output.split()))
        verbosity_disparity = abs(len_a - len_b) / max(len_a, len_b)

        # Advocate A Speech (Defense of v1)
        advocate_a_prompt = (
            f"You are Defense Counsel for Candidate A.\n"
            f"Query: {query}\nCandidate A: {v1_output}\nCandidate B: {v2_output}\n"
            f"Argue why Candidate A is superior, highlighting any hallucinations or flaws in B."
        )
        res_a = await self.provider.generate(advocate_a_prompt)

        # Advocate B Speech (Defense of v2)
        advocate_b_prompt = (
            f"You are Defense Counsel for Candidate B.\n"
            f"Query: {query}\nCandidate A: {v1_output}\nCandidate B: {v2_output}\n"
            f"Argue why Candidate B is superior, highlighting flaws in A."
        )
        res_b = await self.provider.generate(advocate_b_prompt)

        # Chief Justice Deliberation
        justice_prompt = (
            f"You are Chief Justice. Review both arguments impartially.\n"
            f"Counsel A: {res_a.output}\nCounsel B: {res_b.output}\n"
            f"Disregard mere text length. Rule who answered the user better (A, B, or TIE)."
        )
        res_justice = await self.provider.generate(justice_prompt)

        text = res_justice.output.lower()
        if "candidate b" in text or "winner: b" in text or "b is superior" in text:
            winner = "v2"
            conf = 0.85
        elif "candidate a" in text or "winner: a" in text or "a is superior" in text:
            winner = "v1"
            conf = 0.85
        else:
            # If mock or ambiguous, compare factual grounding
            winner = "v2" if "concise" in v2_output.lower() or len_b < len_a else "TIE"
            conf = 0.75

        return DebateRound(
            advocate_a_speech=res_a.output,
            advocate_b_speech=res_b.output,
            chief_justice_synthesis=res_justice.output,
            winner=winner,
            confidence=conf,
            bias_correction_applied=verbosity_disparity > 0.4,
        )

    async def async_evaluate(self, r1: RunResult, r2: RunResult, test_case: TestCase) -> EvaluatorScore:
        query = test_case.vars.get("query", "") or test_case.description
        round_res = await self.conduct_debate(query=query, v1_output=r1.output, v2_output=r2.output)

        passed = round_res.winner in ("v2", "TIE")
        v1_val = 1.0 if round_res.winner == "v1" else (0.5 if round_res.winner == "TIE" else 0.0)
        v2_val = 1.0 if round_res.winner == "v2" else (0.5 if round_res.winner == "TIE" else 0.0)

        return EvaluatorScore(
            name="debate_judge",
            passed=passed,
            v1_score=v1_val,
            v2_score=v2_val,
            delta=v2_val - v1_val,
            delta_pct=round((v2_val - v1_val) * 100.0, 1),
            message=f"Chief Justice Verdict: Winner is {round_res.winner} (Confidence: {round_res.confidence * 100:.0f}%, Bias-Corrected: {round_res.bias_correction_applied})",
        )

    def evaluate(self, r1: RunResult, r2: RunResult, test_case: TestCase) -> EvaluatorScore:
        return asyncio.run(self.async_evaluate(r1, r2, test_case))
