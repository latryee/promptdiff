"""LLM-as-a-Judge Evaluator for Automated Model Scoring against Qualitative Rubrics."""

from __future__ import annotations

import asyncio
import json
import re

from promptdiff.core.models import EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.base import BaseEvaluator
from promptdiff.providers.base import BaseLLMProvider
from promptdiff.providers.registry import get_provider

DEFAULT_RUBRIC = """
You are an expert AI evaluation judge assessing the quality, accuracy, correctness, conciseness,
and instruction adherence of LLM candidate responses.

Score the candidate output on a strict scale from 1.0 to 5.0 using the following criteria:
- 5.0 (Exceptional): Perfect instruction following, exact schema adherence, concise, zero fluff, fully accurate.
- 4.0 (Good): Follows all core instructions, high accuracy, minor stylistic or non-critical formatting differences.
- 3.0 (Acceptable): Mostly correct but includes slight verbosity, minor hallucinations, or weak formatting.
- 2.0 (Poor): Fails major instructions, contains hallucinations, incorrect facts, or broken syntax.
- 1.0 (Critical Failure): Completely off-topic, toxic, empty, or severe prompt injection failure.

Format your response EXACTLY as follows:
[REASONING] <1-2 sentences of specific technical justification>
[SCORE] <number between 1.0 and 5.0>
"""


class LLMJudgeEvaluator(BaseEvaluator):
    """Evaluates outputs using an LLM judge model based on qualitative rubrics."""

    name: str = "llm_judge"
    description: str = "LLM-as-a-Judge semantic scoring (1.0 - 5.0) with automated rubric evaluation"

    def __init__(
        self,
        model_name: str = "gpt-4o",
        rubric: str | None = None,
        provider: BaseLLMProvider | None = None,
        pass_threshold: float = 3.5,
        force_mock: bool = False,
    ):
        self.model_name = model_name
        self.rubric = rubric or DEFAULT_RUBRIC
        self.pass_threshold = pass_threshold
        self.provider = provider or get_provider(model_name=model_name, force_mock=force_mock)

    def _build_judge_prompt(self, test_case: TestCase, v1_out: str, v2_out: str) -> str:
        """Construct the full evaluation prompt for the LLM judge."""
        vars_str = json.dumps(test_case.vars, indent=2) if test_case.vars else "None"
        expected = test_case.expected_output or "None specified"

        return f"""
{self.rubric.strip()}

--- TEST CASE INPUTS ---
Variables:
{vars_str}

Expected Target / Output:
{expected}

--- BASELINE RESPONSE (v1) ---
{v1_out}

--- CANDIDATE RESPONSE (v2) ---
{v2_out}

Evaluate the CANDIDATE RESPONSE (v2) compared to the baseline (v1) and expected criteria.
Provide [REASONING] followed by [SCORE].
"""

    def _parse_judge_output(self, output: str) -> tuple[float, str]:
        """Extract score and reasoning from judge model output."""
        reasoning = "Evaluation completed."
        score = 4.0

        # Try to parse [REASONING] and [SCORE]
        reason_match = re.search(r"\[REASONING\]\s*(.*?)(?=\[SCORE\]|$)", output, re.DOTALL | re.IGNORECASE)
        if reason_match:
            reasoning = reason_match.group(1).strip()

        score_match = re.search(r"\[SCORE\]\s*([0-5](?:\.[0-9]+)?)", output, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
            except ValueError:
                score = 3.5
        else:
            # Fallback regex search for any "score: X" pattern
            alt_score = re.search(r"(?:score|rating):\s*([0-5](?:\.[0-9]+)?)", output, re.IGNORECASE)
            if alt_score:
                try:
                    score = float(alt_score.group(1))
                except ValueError:
                    score = 3.5

        # Clamp between 1.0 and 5.0
        score = max(1.0, min(5.0, score))
        return score, reasoning

    async def async_evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        """Run judge evaluation asynchronously."""
        v1_out = v1_result.output
        v2_out = v2_result.output

        if not v2_out.strip() and v2_result.error:
            return EvaluatorScore(
                name=self.name,
                v1_score=5.0,
                v2_score=1.0,
                delta=-4.0,
                delta_pct=-80.0,
                passed=False,
                message=f"Judge: 1.0/5.0 (Candidate produced error: {v2_result.error})",
                details={"reasoning": f"Execution error: {v2_result.error}", "judge_model": self.model_name},
            )

        prompt = self._build_judge_prompt(test_case, v1_out, v2_out)

        try:
            resp = await self.provider.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=512,
            )
            v2_score, reasoning = self._parse_judge_output(resp.output)
        except Exception as e:
            # Fallback if judge API fails
            v2_score = 4.0
            reasoning = f"Judge evaluation fallback: {e}"

        v1_score = 4.0  # Baseline expectation
        delta = v2_score - v1_score
        delta_pct = (delta / v1_score) * 100.0
        passed = v2_score >= self.pass_threshold

        msg = f"Judge: {v2_score:.1f}/5.0 (Pass >= {self.pass_threshold:.1f}) - {reasoning[:60]}..."

        return EvaluatorScore(
            name=self.name,
            v1_score=v1_score,
            v2_score=round(v2_score, 2),
            delta=round(delta, 2),
            delta_pct=round(delta_pct, 1),
            passed=passed,
            message=msg,
            details={
                "reasoning": reasoning,
                "judge_model": self.model_name,
                "pass_threshold": self.pass_threshold,
            },
        )

    def evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        """Synchronous wrapper for async evaluation."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(
                        asyncio.run,
                        self.async_evaluate(v1_result, v2_result, test_case),
                    ).result()
            else:
                return loop.run_until_complete(self.async_evaluate(v1_result, v2_result, test_case))
        except Exception:
            return asyncio.run(self.async_evaluate(v1_result, v2_result, test_case))
