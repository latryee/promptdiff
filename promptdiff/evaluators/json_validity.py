"""JSON Validity & Schema Evaluator.

Checks whether outputs are valid JSON syntax and optionally adhere to a JSON Schema.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple
from promptdiff.core.models import EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.base import BaseEvaluator


def extract_json(text: str) -> Tuple[Optional[Any], Optional[str]]:
    """Extract and parse JSON from raw text or markdown code block."""
    text_clean = text.strip()

    # 1. Try direct JSON parse
    try:
        return json.loads(text_clean), None
    except Exception as err:
        direct_err = str(err)

    # 2. Try extracting from markdown code block ```json ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text_clean)
    if match:
        try:
            return json.loads(match.group(1)), None
        except Exception as err:
            return None, f"Markdown block JSON parse error: {err}"

    # 3. Try finding first '{' or '[' and matching pair
    start_brace = text_clean.find("{")
    start_bracket = text_clean.find("[")

    start_idx = -1
    end_char = ""
    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        start_idx = start_brace
        end_char = "}"
    elif start_bracket != -1:
        start_idx = start_bracket
        end_char = "]"

    if start_idx != -1:
        end_idx = text_clean.rfind(end_char)
        if end_idx > start_idx:
            substring = text_clean[start_idx : end_idx + 1]
            try:
                return json.loads(substring), None
            except Exception:
                pass

    return None, f"Invalid JSON: {direct_err}"


def validate_schema(data: Any, schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Lightweight built-in JSON Schema validator without heavy third-party deps."""
    if not isinstance(data, dict):
        return False, f"Expected object, got {type(data).__name__}"

    required = schema.get("required", [])
    for field in required:
        if field not in data:
            return False, f"Missing required field: '{field}'"

    properties = schema.get("properties", {})
    type_map = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    for prop, prop_schema in properties.items():
        if prop in data and "type" in prop_schema:
            expected_type_str = prop_schema["type"]
            expected_type = type_map.get(expected_type_str)
            if expected_type and not isinstance(data[prop], expected_type):
                return False, f"Field '{prop}' expected {expected_type_str}, got {type(data[prop]).__name__}"

    return True, None


class JsonValidityEvaluator(BaseEvaluator):
    """Evaluates whether model output is valid JSON and adheres to schema."""

    name: str = "json_validity"
    description: str = "Validates JSON structure and schema compliance (1.0 = Valid, 0.0 = Invalid)"

    def evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        v1_json, v1_err = extract_json(v1_result.output)
        v2_json, v2_err = extract_json(v2_result.output)

        v1_score = 1.0 if v1_json is not None else 0.0
        v2_score = 1.0 if v2_json is not None else 0.0

        v1_schema_err = None
        v2_schema_err = None

        schema = test_case.schema_definition
        if schema:
            if v1_json is not None:
                v1_valid, v1_schema_err = validate_schema(v1_json, schema)
                if not v1_valid:
                    v1_score = 0.5
            if v2_json is not None:
                v2_valid, v2_schema_err = validate_schema(v2_json, schema)
                if not v2_valid:
                    v2_score = 0.5

        delta = v2_score - v1_score
        passed = v2_score >= v1_score

        if v2_score == 1.0:
            msg = "Valid JSON"
        elif v2_schema_err:
            msg = f"Schema mismatch: {v2_schema_err}"
        else:
            msg = f"Invalid JSON: {v2_err}"

        return EvaluatorScore(
            name=self.name,
            v1_score=v1_score,
            v2_score=v2_score,
            delta=round(delta, 2),
            delta_pct=round(delta * 100, 1) if v1_score != 0 else (100.0 if delta > 0 else 0.0),
            passed=passed,
            message=msg,
            details={
                "v1_valid": v1_json is not None,
                "v2_valid": v2_json is not None,
                "v1_error": v1_err or v1_schema_err,
                "v2_error": v2_err or v2_schema_err,
            },
        )
