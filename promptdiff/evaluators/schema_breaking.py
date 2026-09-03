"""Structured Output & JSON Schema Breaking Change Detector.

Detects backward-incompatible structural regressions between prompt output revisions
(e.g., deleted fields, mutated types, enum mutations, nullability violations).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from promptdiff.core.models import EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.base import BaseEvaluator


class SchemaChangeSeverity(str, Enum):
    BREAKING = "BREAKING"
    WARNING = "WARNING"
    COMPATIBLE = "COMPATIBLE"


@dataclass
class SchemaDifference:
    """Individual structural difference between output payloads."""

    path: str
    change_type: str  # REMOVED_FIELD, TYPE_MUTATION, NULLABILITY_VIOLATION, ENUM_VIOLATION
    severity: SchemaChangeSeverity
    v1_value_or_type: Optional[str]
    v2_value_or_type: Optional[str]
    message: str


@dataclass
class SchemaBreakingReport:
    """Evaluation report on structural JSON backward-compatibility."""

    is_compatible: bool
    has_breaking_changes: bool
    breaking_count: int
    warning_count: int
    differences: list[SchemaDifference] = field(default_factory=list)
    summary: str = ""


class SchemaBreakingChangeDetector:
    """Diffs two structured JSON payloads to detect downstream breaking contract changes."""

    def __init__(self, reference_schema: Optional[dict[str, Any]] = None):
        self.reference_schema = reference_schema

    def _get_type_name(self, val: Any) -> str:
        if val is None:
            return "null"
        if isinstance(val, bool):
            return "boolean"
        if isinstance(val, int):
            return "integer"
        if isinstance(val, float):
            return "number"
        if isinstance(val, str):
            return "string"
        if isinstance(val, list):
            return "array"
        if isinstance(val, dict):
            return "object"
        return type(val).__name__

    def compare_payloads(
        self,
        v1_data: Any,
        v2_data: Any,
        current_path: str = "$",
    ) -> list[SchemaDifference]:
        """Recursively inspect structure between baseline v1 and candidate v2."""
        diffs: list[SchemaDifference] = []

        type_v1 = self._get_type_name(v1_data)
        type_v2 = self._get_type_name(v2_data)

        # 1. Type Mutation Check
        if type_v1 != type_v2:
            # Check if nullability changed
            if type_v2 == "null" and type_v1 != "null":
                diffs.append(
                    SchemaDifference(
                        path=current_path,
                        change_type="NULLABILITY_VIOLATION",
                        severity=SchemaChangeSeverity.BREAKING,
                        v1_value_or_type=type_v1,
                        v2_value_or_type="null",
                        message=f"Field '{current_path}' became null in v2 (expected non-null {type_v1}).",
                    )
                )
            else:
                diffs.append(
                    SchemaDifference(
                        path=current_path,
                        change_type="TYPE_MUTATION",
                        severity=SchemaChangeSeverity.BREAKING,
                        v1_value_or_type=type_v1,
                        v2_value_or_type=type_v2,
                        message=f"Type of '{current_path}' mutated from '{type_v1}' to '{type_v2}'.",
                    )
                )
            return diffs

        # 2. Object Key Diffing
        if isinstance(v1_data, dict) and isinstance(v2_data, dict):
            keys_v1 = set(v1_data.keys())
            keys_v2 = set(v2_data.keys())

            # Removed keys are breaking
            removed_keys = keys_v1 - keys_v2
            for k in removed_keys:
                field_path = f"{current_path}.{k}"
                diffs.append(
                    SchemaDifference(
                        path=field_path,
                        change_type="REMOVED_FIELD",
                        severity=SchemaChangeSeverity.BREAKING,
                        v1_value_or_type=self._get_type_name(v1_data[k]),
                        v2_value_or_type=None,
                        message=f"Required field '{field_path}' was omitted in candidate v2 response.",
                    )
                )

            # Added keys are non-breaking extensions (WARNING/COMPATIBLE)
            added_keys = keys_v2 - keys_v1
            for k in added_keys:
                field_path = f"{current_path}.{k}"
                diffs.append(
                    SchemaDifference(
                        path=field_path,
                        change_type="ADDED_FIELD",
                        severity=SchemaChangeSeverity.WARNING,
                        v1_value_or_type=None,
                        v2_value_or_type=self._get_type_name(v2_data[k]),
                        message=f"New field '{field_path}' introduced in candidate v2 response.",
                    )
                )

            # Recurse on shared keys
            for k in keys_v1 & keys_v2:
                field_path = f"{current_path}.{k}"
                diffs.extend(self.compare_payloads(v1_data[k], v2_data[k], current_path=field_path))

        # 3. Array Element Diffing
        elif isinstance(v1_data, list) and isinstance(v2_data, list):
            if v1_data and v2_data:
                # Sample compare first element types
                elem_path = f"{current_path}[0]"
                diffs.extend(self.compare_payloads(v1_data[0], v2_data[0], current_path=elem_path))
            elif v1_data and not v2_data:
                diffs.append(
                    SchemaDifference(
                        path=current_path,
                        change_type="EMPTY_ARRAY",
                        severity=SchemaChangeSeverity.WARNING,
                        v1_value_or_type="non-empty array",
                        v2_value_or_type="empty array",
                        message=f"Array at '{current_path}' was populated in v1 but empty in v2.",
                    )
                )

        return diffs

    def evaluate(self, v1_raw: str, v2_raw: str) -> SchemaBreakingReport:
        """Evaluate two raw JSON string responses."""
        try:
            data_v1 = json.loads(v1_raw)
        except Exception:
            return SchemaBreakingReport(
                is_compatible=False,
                has_breaking_changes=True,
                breaking_count=1,
                warning_count=0,
                differences=[
                    SchemaDifference(
                        path="$",
                        change_type="PARSE_ERROR_V1",
                        severity=SchemaChangeSeverity.BREAKING,
                        v1_value_or_type="invalid_json",
                        v2_value_or_type=None,
                        message="v1 baseline output is not valid JSON.",
                    )
                ],
                summary="Baseline response is not valid JSON.",
            )

        try:
            data_v2 = json.loads(v2_raw)
        except Exception:
            return SchemaBreakingReport(
                is_compatible=False,
                has_breaking_changes=True,
                breaking_count=1,
                warning_count=0,
                differences=[
                    SchemaDifference(
                        path="$",
                        change_type="PARSE_ERROR_V2",
                        severity=SchemaChangeSeverity.BREAKING,
                        v1_value_or_type="valid_json",
                        v2_value_or_type="invalid_json",
                        message="v2 candidate output is not valid JSON.",
                    )
                ],
                summary="Candidate response failed JSON parsing.",
            )

        diffs = self.compare_payloads(data_v1, data_v2)
        breaking = [d for d in diffs if d.severity == SchemaChangeSeverity.BREAKING]
        warnings = [d for d in diffs if d.severity == SchemaChangeSeverity.WARNING]

        has_breaking = len(breaking) > 0
        summary = (
            f"Found {len(breaking)} breaking change(s) and {len(warnings)} warning(s)."
            if has_breaking
            else "Schema backward-compatible. No breaking structural changes detected."
        )

        return SchemaBreakingReport(
            is_compatible=not has_breaking,
            has_breaking_changes=has_breaking,
            breaking_count=len(breaking),
            warning_count=len(warnings),
            differences=diffs,
            summary=summary,
        )


class SchemaBreakingChangeEvaluator(BaseEvaluator):
    """PromptDiff evaluator that penalizes schema contract breaking changes."""

    name = "schema_breaking"

    def __init__(self, reference_schema: Optional[dict[str, Any]] = None):
        super().__init__()
        self.detector = SchemaBreakingChangeDetector(reference_schema=reference_schema)

    def evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        report = self.detector.evaluate(v1_result.output, v2_result.output)

        score = 1.0 if not report.has_breaking_changes else max(0.0, 1.0 - (report.breaking_count * 0.3))
        passed = not report.has_breaking_changes

        details = {
            "is_compatible": report.is_compatible,
            "breaking_count": report.breaking_count,
            "warning_count": report.warning_count,
            "differences": [
                {
                    "path": d.path,
                    "change_type": d.change_type,
                    "severity": d.severity.value,
                    "message": d.message,
                }
                for d in report.differences
            ],
        }

        return EvaluatorScore(
            name=self.name,
            v1_score=1.0,
            v2_score=round(score, 2),
            passed=passed,
            message=report.summary,
            details=details,
        )


def detect_schema_breaking_changes(v1_json: str, v2_json: str) -> SchemaBreakingReport:
    """Analyze two JSON string payloads for backward-incompatible structural changes."""
    detector = SchemaBreakingChangeDetector()
    return detector.evaluate(v1_json, v2_json)
