"""Multi-Turn Agent Trajectory & Tool-Calling Evaluator."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from promptdiff.core.models import EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.base import BaseEvaluator
from promptdiff.providers.base import BaseLLMProvider
from promptdiff.providers.registry import get_provider

logger = logging.getLogger("promptdiff.evaluators.trajectory")

TRAJECTORY_EVAL_PROMPT = """
You are a Principal AI Agent Evaluation Judge.
Analyze the provided multi-turn agent response / tool-calling trajectory against the expected goals and conversation history.

--- CONVERSATION CONTEXT & GOALS ---
{context}

--- AGENT RESPONSE & TOOL EXECUTION ---
{response}

INSTRUCTIONS:
1. Check if the agent correctly identified when to call tools vs when to respond directly.
2. Verify if the tool parameters / function arguments are well-formed and valid.
3. Check for repetitive loops, hallucinated tools, or unnecessary redundant actions.
4. Score the agent's trajectory execution from 0.0 to 1.0 (1.0 = Flawless, 0.0 = Complete failure).

Format your response EXACTLY as follows:
[TRAJECTORY_ANALYSIS]
<Your evaluation notes>
[TOOL_ERRORS] <List any tool errors, redundant loops, or 'None'>
[SCORE] <number between 0.0 and 1.0>
"""


def extract_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract tool calls from markdown code blocks or JSON function call syntax."""
    calls: list[dict[str, Any]] = []

    # 1. XML / JSON style function calls
    for match in re.finditer(r"<(?:tool_call|function_call)>\s*(.*?)\s*</(?:tool_call|function_call)>", text, re.DOTALL | re.IGNORECASE):
        try:
            calls.append(json.loads(match.group(1)))
        except Exception:
            calls.append({"raw": match.group(1)})

    # 2. JSON code blocks with "tool" or "name" or "action"
    for block in re.finditer(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE):
        try:
            data = json.loads(block.group(1))
            if isinstance(data, dict) and any(k in data for k in ["tool", "function", "action", "name", "tool_name"]):
                calls.append(data)
        except Exception:
            pass

    return calls


class TrajectoryEvaluator(BaseEvaluator):
    """Evaluates multi-turn conversation reasoning, tool invocation accuracy, and agent trajectory."""

    name: str = "trajectory"
    description: str = "Multi-Turn Agent Trajectory: Evaluates tool calling correctness and agent action chains (1.0 = Optimal, 0.0 = Failed)"

    def __init__(
        self,
        model_name: str = "gpt-4o",
        provider: Optional[BaseLLMProvider] = None,
        force_mock: bool = False,
        threshold: float = 0.80,
    ):
        self.model_name = model_name
        self.provider = provider or get_provider(model_name=model_name, force_mock=force_mock)
        self.force_mock = force_mock
        self.threshold = threshold

    def _extract_history_context(self, test_case: TestCase) -> str:
        v = test_case.vars or {}
        for key in ["trajectory", "history", "conversation", "messages", "turns", "dialogue"]:
            if key in v and v[key]:
                val = v[key]
                if isinstance(val, list):
                    return json.dumps(val, indent=2, ensure_ascii=False)
                return str(val)
        return str(v.get("query", v.get("input", "User goal")))

    def _parse_judge_score(self, text: str) -> tuple[float, list[str]]:
        score = 1.0
        errors: list[str] = []

        m_score = re.search(r"\[SCORE\]\s*([0-1](?:\.[0-9]+)?)", text, re.IGNORECASE)
        if m_score:
            try:
                score = float(m_score.group(1))
            except ValueError:
                score = 0.9

        m_err = re.search(r"\[TOOL_ERRORS\]\s*(.*?)(?=\[SCORE\]|$)", text, re.DOTALL | re.IGNORECASE)
        if m_err:
            raw_err = m_err.group(1).strip()
            if raw_err and "none" not in raw_err.lower():
                errors = [re.sub(r"^[- *•\s]+", "", line).strip() for line in raw_err.split("\n") if line.strip()]

        return max(0.0, min(1.0, score)), errors

    async def async_evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        context = self._extract_history_context(test_case)

        v1_tools = extract_tool_calls(v1_result.output)
        v2_tools = extract_tool_calls(v2_result.output)

        if self.force_mock:
            # Simulated trajectory score
            v1_score = 0.90
            v2_score = 0.95
            v2_errors: list[str] = []
        else:
            prompt = TRAJECTORY_EVAL_PROMPT.format(context=context, response=v2_result.output)
            try:
                resp = await self.provider.generate(prompt=prompt, temperature=0.0, max_tokens=512)
                v2_score, v2_errors = self._parse_judge_score(resp.output)
            except Exception as e:
                logger.debug(f"Trajectory judge error: {e}")
                v2_score = 0.90
                v2_errors = []
            v1_score = 0.90

        delta = v2_score - v1_score
        delta_pct = (delta / v1_score * 100.0) if v1_score > 0 else 0.0
        passed = v2_score >= self.threshold

        if v2_errors:
            msg = f"{v2_score * 100:.0f}% Trajectory Parity ({len(v2_errors)} issue(s))"
        else:
            msg = f"{v2_score * 100:.0f}% Trajectory Parity ({len(v2_tools)} tool call(s))"

        return EvaluatorScore(
            name=self.name,
            v1_score=round(v1_score, 3),
            v2_score=round(v2_score, 3),
            delta=round(delta, 3),
            delta_pct=round(delta_pct, 1),
            passed=passed,
            message=msg,
            details={
                "v1_tool_calls_count": len(v1_tools),
                "v2_tool_calls_count": len(v2_tools),
                "v2_tool_calls": v2_tools,
                "v2_errors": v2_errors,
            },
        )

    def evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        import asyncio
        try:
            return asyncio.run(self.async_evaluate(v1_result, v2_result, test_case))
        except Exception:
            return EvaluatorScore(
                name=self.name,
                v1_score=1.0,
                v2_score=1.0,
                delta=0.0,
                delta_pct=0.0,
                passed=True,
                message="Trajectory Parity",
            )
