"""Domain Models and Type Definitions for promptdiff.

Structured with strict Pydantic v2 validation and typed schemas.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class TestCase(BaseModel):
    """Represents a single input scenario for prompt regression testing."""

    __test__ = False

    id: str = Field(default_factory=lambda: f"test_{uuid.uuid4().hex[:6]}")
    description: str = ""
    vars: Dict[str, Any] = Field(default_factory=dict)
    expected_output: Optional[str] = None
    schema_definition: Optional[Dict[str, Any]] = Field(default=None, alias="schema")
    tags: List[str] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)


class PromptVersion(BaseModel):
    """Represents a prompt template version with execution metadata."""

    name: str = "v1"
    path: Optional[str] = None
    template: str
    model: str = "gpt-4o"
    temperature: float = 0.0
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = 2048

    def render(self, variables: Dict[str, Any]) -> str:
        """Render prompt template with variable substitution.

        Supports {{var_name}}, {var_name}, and Jinja2-style syntax.
        """
        text = self.template
        for key, value in variables.items():
            # Double braces: {{key}}
            text = text.replace(f"{{{{{key}}}}}", str(value))
            # Single braces: {key} (if not already part of double brace)
            text = re.sub(rf"(?<!\{{)\{{{re.escape(key)}\}}(?!\}})", str(value), text)
        return text


class RunResult(BaseModel):
    """Output and performance telemetry from an LLM run."""

    prompt_name: str
    test_case_id: str
    rendered_prompt: str
    output: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    model: str
    cached: bool = False
    error: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


class EvaluatorScore(BaseModel):
    """Result of an evaluation metric comparison."""

    name: str
    v1_score: Any
    v2_score: Any
    delta: Optional[float] = None
    delta_pct: Optional[float] = None
    passed: bool = True
    message: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)


class DiffChunk(BaseModel):
    """A diff chunk between two output strings."""

    kind: Literal["equal", "insert", "delete", "replace"]
    v1_text: str = ""
    v2_text: str = ""


class ComparisonResult(BaseModel):
    """Side-by-side comparison of two prompt version executions."""

    test_case: TestCase
    v1_result: RunResult
    v2_result: RunResult
    scores: Dict[str, EvaluatorScore] = Field(default_factory=dict)
    text_diff: List[DiffChunk] = Field(default_factory=list)
    json_diff: Optional[Dict[str, Any]] = None
    is_json: bool = False


class AssertionRule(BaseModel):
    """Rule defining a pass/fail threshold for CI/CD regression checks."""

    metric: str
    operator: Literal["<=", ">=", "<", ">", "==", "!="]
    threshold: float
    is_pct: bool = False
    raw_expression: str = ""

    def evaluate(self, actual_value: float) -> bool:
        """Evaluate actual value against assertion threshold."""
        op = self.operator
        t = self.threshold
        if op == "<=":
            return actual_value <= t
        elif op == ">=":
            return actual_value >= t
        elif op == "<":
            return actual_value < t
        elif op == ">":
            return actual_value > t
        elif op == "==":
            return abs(actual_value - t) < 1e-6
        elif op == "!=":
            return abs(actual_value - t) >= 1e-6
        return False


class RegressionVerdict(BaseModel):
    """Overall run status and list of failed assertions."""

    passed: bool = True
    status: Literal["PASSED", "REGRESSION_DETECTED", "ERROR"] = "PASSED"
    failed_assertions: List[str] = Field(default_factory=list)
    total_cost_v1: float = 0.0
    total_cost_v2: float = 0.0
    cost_delta_pct: float = 0.0
    avg_latency_v1: float = 0.0
    avg_latency_v2: float = 0.0
    latency_delta_pct: float = 0.0
    summary_metrics: Dict[str, Any] = Field(default_factory=dict)


class DiffReport(BaseModel):
    """Complete serialized evaluation and regression report."""

    run_id: str = Field(default_factory=lambda: f"run_{uuid.uuid4().hex[:8]}")
    timestamp: str = ""
    v1_name: str
    v2_name: str
    model_v1: str
    model_v2: str
    comparisons: List[ComparisonResult]
    verdict: RegressionVerdict
    evaluators: List[str]
    total_cases: int
    aggregate_stats: Dict[str, Any] = Field(default_factory=dict)
