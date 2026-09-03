"""RAG Faithfulness & Hallucination Detection Evaluator.

Measures whether generated outputs are strictly grounded in and entailed by provided context documents,
detecting ungrounded claims, hallucinations, and extrapolation.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

from promptdiff.core.models import EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.base import BaseEvaluator
from promptdiff.providers.base import BaseLLMProvider
from promptdiff.providers.registry import get_provider

logger = logging.getLogger("promptdiff.evaluators.faithfulness")

FAITHFULNESS_PROMPT = """
You are a strict RAG Factual Grounding & Hallucination Detection Judge.
Your task is to analyze the provided RESPONSE and determine whether ALL factual claims made in the response
are directly grounded in and supported by the reference CONTEXT.

--- CONTEXT ---
{context}

--- RESPONSE ---
{response}

INSTRUCTIONS:
1. Extract the core factual claims or statements made in the RESPONSE.
2. For each claim, check if it is directly stated in or logically entailed by the CONTEXT.
3. If a claim contains information NOT in the context, mark it as UNGROUNDED / HALLUCINATION.
4. Calculate the faithfulness score from 0.0 to 1.0 (Grounded Claims / Total Claims). If response has zero factual claims or says 'I do not have enough context', score is 1.0.

Format your response EXACTLY as follows:
[CLAIMS_EVALUATION]
- Claim 1: [GROUNDED or UNGROUNDED] - <explanation>
[HALLUCINATIONS] <List any ungrounded claims, or 'None'>
[SCORE] <number between 0.0 and 1.0>
"""


class FaithfulnessEvaluator(BaseEvaluator):
    """Evaluates factual consistency and groundedness against retrieval context."""

    name: str = "faithfulness"
    description: str = (
        "RAG Groundedness & Hallucination check against provided context (1.0 = Fully Faithful, 0.0 = Hallucinated)"
    )

    def __init__(
        self,
        model_name: str = "gpt-4o",
        provider: Optional[BaseLLMProvider] = None,
        threshold: float = 0.80,
        force_mock: bool = False,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.provider = provider or get_provider(model_name=model_name, force_mock=force_mock)

    def _extract_context(self, test_case: TestCase) -> Optional[str]:
        """Extract context string from testcase variables."""
        vars_dict = test_case.vars or {}
        for key in ["context", "docs", "retrieved_context", "knowledge", "reference", "documents"]:
            if key in vars_dict and vars_dict[key]:
                val = vars_dict[key]
                if isinstance(val, (list, tuple)):
                    return "\n\n".join(str(item) for item in val)
                return str(val)
        return None

    def _parse_faithfulness_output(self, output: str) -> tuple[float, list[str]]:
        """Parse faithfulness score and hallucination list from judge response."""
        score = 1.0
        hallucinations: list[str] = []

        score_match = re.search(r"\[SCORE\]\s*([0-1](?:\.[0-9]+)?)", output, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
            except ValueError:
                score = 0.8
        else:
            alt_match = re.search(r"(?:faithfulness|score):\s*([0-1](?:\.[0-9]+)?)", output, re.IGNORECASE)
            if alt_match:
                try:
                    score = float(alt_match.group(1))
                except ValueError:
                    score = 0.8

        hallu_match = re.search(r"\[HALLUCINATIONS\]\s*(.*?)(?=\[SCORE\]|$)", output, re.DOTALL | re.IGNORECASE)
        if hallu_match:
            raw_h = hallu_match.group(1).strip()
            if raw_h and "none" not in raw_h.lower():
                hallucinations = [re.sub(r"^[- *•\s]+", "", line).strip() for line in raw_h.split("\n") if line.strip()]

        return max(0.0, min(1.0, score)), hallucinations

    def _heuristic_check(self, context: str, response: str) -> tuple[float, list[str]]:
        """Fast heuristic check for mock / offline fallback."""
        words_ctx = set(re.findall(r"\w{4,}", context.lower()))
        words_resp = set(re.findall(r"\w{4,}", response.lower()))
        if not words_resp:
            return 1.0, []

        overlap = len(words_resp.intersection(words_ctx)) / len(words_resp)
        score = min(1.0, overlap * 1.3)
        return round(score, 2), []

    async def _evaluate_output(self, context: str, output: str) -> tuple[float, list[str]]:
        """Evaluate a single output string against reference context."""
        if not output.strip():
            return 1.0, []

        prompt = FAITHFULNESS_PROMPT.format(context=context, response=output)
        try:
            resp = await self.provider.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=512,
            )
            return self._parse_faithfulness_output(resp.output)
        except Exception as e:
            logger.debug(f"Faithfulness LLM check fallback to heuristic: {e}")
            return self._heuristic_check(context, output)

    async def async_evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        context = self._extract_context(test_case)

        if not context:
            return EvaluatorScore(
                name=self.name,
                v1_score=1.0,
                v2_score=1.0,
                delta=0.0,
                delta_pct=0.0,
                passed=True,
                message="Faithful (No reference context in testcase)",
                details={"context_provided": False},
            )

        v1_score, v1_hallucinations = await self._evaluate_output(context, v1_result.output)
        v2_score, v2_hallucinations = await self._evaluate_output(context, v2_result.output)

        delta = v2_score - v1_score
        delta_pct = (delta / v1_score * 100.0) if v1_score > 0 else 0.0
        passed = v2_score >= self.threshold

        if v2_hallucinations:
            msg = f"{v2_score * 100:.1f}% Grounded ({len(v2_hallucinations)} hallucination(s) detected)"
        else:
            msg = f"{v2_score * 100:.1f}% Grounded (Zero hallucinations)"

        return EvaluatorScore(
            name=self.name,
            v1_score=round(v1_score, 3),
            v2_score=round(v2_score, 3),
            delta=round(delta, 3),
            delta_pct=round(delta_pct, 1),
            passed=passed,
            message=msg,
            details={
                "context_provided": True,
                "v1_faithfulness": v1_score,
                "v2_faithfulness": v2_score,
                "v2_hallucinations": v2_hallucinations,
                "threshold": self.threshold,
            },
        )

    def evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
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
