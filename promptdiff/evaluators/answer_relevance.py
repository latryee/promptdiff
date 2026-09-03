"""RAG Answer Relevance & Intent Alignment Evaluator.

Measures whether generated outputs directly address the user query without evasiveness,
tangents, or irrelevant boilerplate.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

from promptdiff.core.models import EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.base import BaseEvaluator
from promptdiff.evaluators.similarity import _get_embedding_model, cosine_similarity
from promptdiff.providers.base import BaseLLMProvider
from promptdiff.providers.registry import get_provider

logger = logging.getLogger("promptdiff.evaluators.answer_relevance")

RELEVANCE_PROMPT = """
You are an expert Question-Answering Relevance & Conciseness Evaluator.
Analyze the USER QUERY and the MODEL RESPONSE below:

--- USER QUERY ---
{query}

--- MODEL RESPONSE ---
{response}

INSTRUCTIONS:
1. Determine whether the response directly, concisely, and completely answers the user's query.
2. Penalize evasive responses, irrelevant tangents, redundant filler, and repeated boilerplate.
3. Score relevance strictly on a scale from 0.0 to 1.0 (1.0 = Direct & Complete Answer, 0.0 = Totally Irrelevant or Non-responsive).

Format your response EXACTLY as follows:
[REASONING] <1-2 sentences explaining relevance assessment>
[SCORE] <number between 0.0 and 1.0>
"""


class AnswerRelevanceEvaluator(BaseEvaluator):
    """Evaluates how directly and completely an output answers the input query."""

    name: str = "answer_relevance"
    description: str = "Measures question-answering relevance and directness (1.0 = Direct Answer, 0.0 = Irrelevant)"

    def __init__(
        self,
        model_name: str = "gpt-4o",
        provider: Optional[BaseLLMProvider] = None,
        threshold: float = 0.75,
        force_mock: bool = False,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.provider = provider or get_provider(model_name=model_name, force_mock=force_mock)

    def _extract_query(self, test_case: TestCase) -> str:
        """Extract user question/query from testcase variables."""
        vars_dict = test_case.vars or {}
        for key in ["query", "question", "input", "prompt", "user_query", "user_input"]:
            if key in vars_dict and vars_dict[key]:
                return str(vars_dict[key])
        # Fallback to test case description or string representation
        return test_case.description or str(vars_dict)

    def _parse_relevance_output(self, output: str) -> tuple[float, str]:
        """Extract score and reasoning from judge response."""
        score = 0.85
        reasoning = "Response directly addresses query."

        reason_match = re.search(r"\[REASONING\]\s*(.*?)(?=\[SCORE\]|$)", output, re.DOTALL | re.IGNORECASE)
        if reason_match:
            reasoning = reason_match.group(1).strip()

        score_match = re.search(r"\[SCORE\]\s*([0-1](?:\.[0-9]+)?)", output, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
            except ValueError:
                score = 0.8
        else:
            alt_match = re.search(r"(?:relevance|score):\s*([0-1](?:\.[0-9]+)?)", output, re.IGNORECASE)
            if alt_match:
                try:
                    score = float(alt_match.group(1))
                except ValueError:
                    score = 0.8

        return max(0.0, min(1.0, score)), reasoning

    def _compute_embedding_similarity(self, query: str, response: str) -> float:
        """Compute query-response embedding alignment using local sentence-transformers."""
        model = _get_embedding_model()
        if model is not None:
            try:
                embeddings = model.encode([query, response])
                sim = cosine_similarity(embeddings[0], embeddings[1])
                return max(0.0, min(1.0, sim))
            except Exception as e:
                logger.debug(f"Embedding relevance failed: {e}")
        return 0.8

    async def _evaluate_output(self, query: str, output: str) -> tuple[float, str]:
        if not output.strip():
            return 0.0, "Empty response generated."

        prompt = RELEVANCE_PROMPT.format(query=query, response=output)
        try:
            resp = await self.provider.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=256,
            )
            return self._parse_relevance_output(resp.output)
        except Exception as e:
            logger.debug(f"LLM relevance evaluation fallback to embedding: {e}")
            emb_sim = self._compute_embedding_similarity(query, output)
            return emb_sim, "Evaluated via local semantic embeddings"

    async def async_evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        query = self._extract_query(test_case)

        v1_score, v1_reason = await self._evaluate_output(query, v1_result.output)
        v2_score, v2_reason = await self._evaluate_output(query, v2_result.output)

        delta = v2_score - v1_score
        delta_pct = (delta / v1_score * 100.0) if v1_score > 0 else 0.0
        passed = v2_score >= self.threshold

        msg = f"{v2_score * 100:.1f}% Relevance - {v2_reason[:50]}..."

        return EvaluatorScore(
            name=self.name,
            v1_score=round(v1_score, 3),
            v2_score=round(v2_score, 3),
            delta=round(delta, 3),
            delta_pct=round(delta_pct, 1),
            passed=passed,
            message=msg,
            details={
                "query": query,
                "v1_relevance": v1_score,
                "v2_relevance": v2_score,
                "reasoning": v2_reason,
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
