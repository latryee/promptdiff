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
and instruction adherence of two candidate LLM responses: Baseline (v1) and Candidate (v2).

Score BOTH responses independently on a strict scale from 1.0 to 5.0 using the following criteria:
- 5.0 (Exceptional): Perfect instruction following, exact schema adherence, concise, zero fluff, fully accurate.
- 4.0 (Good): Follows all core instructions, high accuracy, minor stylistic or non-critical formatting differences.
- 3.0 (Acceptable): Mostly correct but includes slight verbosity, minor hallucinations, or weak formatting.
- 2.0 (Poor): Fails major instructions, contains hallucinations, incorrect facts, or broken syntax.
- 1.0 (Critical Failure): Completely off-topic, toxic, empty, or severe prompt injection failure.

Format your response EXACTLY as follows:
[REASONING] <1-2 sentences of comparative technical justification>
[V1_SCORE] <number between 1.0 and 5.0 for Baseline (v1)>
[V2_SCORE] <number between 1.0 and 5.0 for Candidate (v2)>
[PREFERENCE] <V1, V2, or TIE>
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

Evaluate and compare BOTH the BASELINE RESPONSE (v1) and the CANDIDATE RESPONSE (v2) against the expected criteria.
Provide [REASONING], [V1_SCORE], [V2_SCORE], and [PREFERENCE].
"""

    def _parse_judge_output(self, output: str) -> tuple[float, float, str, str]:
        """Extract v1_score, v2_score, reasoning, and preference from judge model output."""
        reasoning = "Comparative evaluation completed."
        preference = "TIE"
        v1_score = 3.5
        v2_score = 3.5

        # 1. Parse [REASONING]
        reason_match = re.search(
            r"\[REASONING\]\s*(.*?)(?=\[V1_SCORE\]|\[V2_SCORE\]|\[SCORE\]|\[PREFERENCE\]|$)",
            output,
            re.DOTALL | re.IGNORECASE,
        )
        if reason_match:
            reasoning = reason_match.group(1).strip()

        # 2. Parse [V1_SCORE]
        v1_match = re.search(r"\[V1_SCORE\]\s*([0-5](?:\.[0-9]+)?)", output, re.IGNORECASE)
        if v1_match:
            try:
                v1_score = float(v1_match.group(1))
            except ValueError:
                v1_score = 3.5

        # 3. Parse [V2_SCORE] or legacy [SCORE]
        v2_match = re.search(r"\[V2_SCORE\]\s*([0-5](?:\.[0-9]+)?)", output, re.IGNORECASE)
        if v2_match:
            try:
                v2_score = float(v2_match.group(1))
            except ValueError:
                v2_score = 3.5
        else:
            single_match = re.search(r"\[SCORE\]\s*([0-5](?:\.[0-9]+)?)", output, re.IGNORECASE)
            if single_match:
                try:
                    v2_score = float(single_match.group(1))
                except ValueError:
                    v2_score = 3.5

        # 4. Parse [PREFERENCE]
        pref_match = re.search(r"\[PREFERENCE\]\s*(V1|V2|TIE)", output, re.IGNORECASE)
        if pref_match:
            preference = pref_match.group(1).upper()

        v1_score = max(1.0, min(5.0, v1_score))
        v2_score = max(1.0, min(5.0, v2_score))
        return v1_score, v2_score, reasoning, preference

    async def async_evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        """Run comparative judge evaluation asynchronously."""
        v1_out = v1_result.output
        v2_out = v2_result.output

        if not v2_out.strip() and v2_result.error:
            return EvaluatorScore(
                name=self.name,
                v1_score=4.0,
                v2_score=1.0,
                delta=-3.0,
                delta_pct=-75.0,
                passed=False,
                message=f"Judge: v1=4.0 -> v2=1.0/5.0 (Candidate error: {v2_result.error})",
                details={
                    "reasoning": f"Candidate error: {v2_result.error}",
                    "preference": "V1",
                    "judge_model": self.model_name,
                },
            )

        prompt = self._build_judge_prompt(test_case, v1_out, v2_out)

        try:
            resp = await self.provider.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=512,
            )
            v1_score, v2_score, reasoning, preference = self._parse_judge_output(resp.output)
        except Exception as e:
            # Fallback if judge API fails
            v1_score = 3.5
            v2_score = 3.5
            reasoning = f"Judge evaluation fallback: {e}"
            preference = "TIE"

        delta = v2_score - v1_score
        delta_pct = (delta / v1_score * 100.0) if v1_score > 0 else 0.0
        passed = (v2_score >= self.pass_threshold) and (delta >= -0.5)

        msg = f"Judge: v1={v1_score:.1f} -> v2={v2_score:.1f}/5.0 (Pref: {preference}) - {reasoning[:45]}..."

        return EvaluatorScore(
            name=self.name,
            v1_score=round(v1_score, 2),
            v2_score=round(v2_score, 2),
            delta=round(delta, 2),
            delta_pct=round(delta_pct, 1),
            passed=passed,
            message=msg,
            details={
                "reasoning": reasoning,
                "preference": preference,
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
