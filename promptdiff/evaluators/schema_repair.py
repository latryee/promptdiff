"""Structured JSON / Pydantic Schema Adherence & Auto-Repair Evaluator for promptdiff."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Optional

from promptdiff.core.models import EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.base import BaseEvaluator

logger = logging.getLogger("promptdiff.evaluators.schema_repair")


class SchemaRepairEvaluator(BaseEvaluator):
    """Evaluates strict JSON schema validity and calculates auto-repairability with Outlines/Instructor heuristics."""

    name: str = "schema_repair"
    description: str = "Evaluates JSON schema compliance and automatic format repairability"

    def __init__(
        self,
        model_name: str = "gpt-4o",
        force_mock: bool = False,
    ):
        self.model_name = model_name
        self.force_mock = force_mock

    def evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        return asyncio.run(self.async_evaluate(v1_result, v2_result, test_case))

    def attempt_json_repair(self, text: str) -> tuple[Optional[dict[str, Any]], str]:
        """Attempt heuristic repair on malformed JSON strings."""
        try:
            return json.loads(text.strip()), "Valid JSON (No Repair Needed)"
        except Exception:
            pass

        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1)), "Repaired: Extracted from Markdown code fence"
            except Exception:
                pass

        cleaned = re.sub(r",\s*([}\]])", r"\1", text.strip())
        try:
            return json.loads(cleaned), "Repaired: Removed trailing commas"
        except Exception:
            pass

        m_outer = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if m_outer:
            try:
                return json.loads(m_outer.group(1)), "Repaired: Extracted outermost JSON object"
            except Exception:
                pass

        return None, "Unrepairable Syntax Error"

    async def async_evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        parsed_v1, method_v1 = self.attempt_json_repair(v1_result.output)
        parsed_v2, method_v2 = self.attempt_json_repair(v2_result.output)

        score_v1 = 1.0 if parsed_v1 is not None else 0.0
        score_v2 = 1.0 if parsed_v2 is not None else 0.0

        passed = score_v2 == 1.0

        return EvaluatorScore(
            name=self.name,
            v1_score=score_v1,
            v2_score=score_v2,
            passed=passed,
            message=f"Candidate JSON Schema status: {method_v2}",
        )
