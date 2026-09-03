"""Domain Models and Type Definitions for promptdiff v2.0.

Structured with strict Pydantic v2 validation and typed schemas.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TestCase(BaseModel):
    """Represents a single input scenario for prompt regression testing."""

    __test__ = False

    id: str = Field(default_factory=lambda: f"test_{uuid.uuid4().hex[:6]}")
    description: str = ""
    vars: dict[str, Any] = Field(default_factory=dict)
    expected_output: str | None = None
    schema_definition: dict[str, Any] | None = Field(default=None, alias="schema")
    tags: list[str] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)


class PromptVersion(BaseModel):
    """Represents a prompt template version with execution metadata."""

    name: str = "v1"
    path: str | None = None
    template: str
    model: str = "gpt-4o"
    temperature: float = 0.0
    system_prompt: str | None = None
    max_tokens: int | None = 2048

    def render(self, variables: dict[str, Any]) -> str:
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


class ConversationVersion(PromptVersion):
    """Multi-turn conversation version consisting of ordered message turns."""

    template: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)

    def render_messages(self, variables: dict[str, Any]) -> list[dict[str, Any]]:
        """Render all messages in the conversation template with variable substitution."""
        rendered: list[dict[str, Any]] = []
        for msg in self.messages:
            content = str(msg.get("content", ""))
            for key, value in variables.items():
                content = content.replace(f"{{{{{key}}}}}", str(value))
                content = re.sub(rf"(?<!\{{)\{{{re.escape(key)}\}}(?!\}})", str(value), content)
            rendered.append({**msg, "content": content})
        return rendered

    def render(self, variables: dict[str, Any]) -> str:
        """Render multi-turn conversation into a unified formatted dialogue transcript."""
        rendered = self.render_messages(variables)
        parts: list[str] = []
        for m in rendered:
            role = str(m.get("role", "user")).capitalize()
            parts.append(f"{role}: {m.get('content', '')}")
        return "\n".join(parts)


class RunResult(BaseModel):
    """Output and performance telemetry from an LLM run."""

    model_config = ConfigDict(frozen=True)

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
    error: str | None = None
    timestamp: float = Field(default_factory=time.time)


class EvaluatorScore(BaseModel):
    """Result of an evaluation metric comparison."""

    model_config = ConfigDict(frozen=True)

    name: str
    v1_score: Any
    v2_score: Any
    delta: float | None = None
    delta_pct: float | None = None
    passed: bool = True
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class DiffChunk(BaseModel):
    """A diff chunk between two output strings."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["equal", "insert", "delete", "replace"]
    v1_text: str = ""
    v2_text: str = ""


class ComparisonResult(BaseModel):
    """Side-by-side comparison of two prompt version executions."""

    model_config = ConfigDict(frozen=True)

    test_case: TestCase
    v1_result: RunResult
    v2_result: RunResult
    scores: dict[str, EvaluatorScore] = Field(default_factory=dict)
    text_diff: list[DiffChunk] = Field(default_factory=list)
    json_diff: dict[str, Any] | None = None
    is_json: bool = False


class MultiComparisonResult(BaseModel):
    """Comparison across N arbitrary prompt/model variants."""

    model_config = ConfigDict(frozen=True)

    test_case: TestCase
    results: dict[str, RunResult] = Field(default_factory=dict)
    scores: dict[str, dict[str, EvaluatorScore]] = Field(default_factory=dict)
    pairwise_diffs: dict[str, list[DiffChunk]] = Field(default_factory=dict)


class AssertionRule(BaseModel):
    """Rule defining a pass/fail threshold for CI/CD regression checks."""

    model_config = ConfigDict(frozen=True)

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

    model_config = ConfigDict(frozen=True)

    passed: bool = True
    status: Literal["PASSED", "REGRESSION_DETECTED", "ERROR"] = "PASSED"
    failed_assertions: list[str] = Field(default_factory=list)
    total_cost_v1: float = 0.0
    total_cost_v2: float = 0.0
    cost_delta_pct: float = 0.0
    avg_latency_v1: float = 0.0
    avg_latency_v2: float = 0.0
    latency_delta_pct: float = 0.0
    summary_metrics: dict[str, Any] = Field(default_factory=dict)


class DiffReport(BaseModel):
    """Complete serialized evaluation and regression report."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(default_factory=lambda: f"run_{uuid.uuid4().hex[:8]}")
    timestamp: str = ""
    v1_name: str
    v2_name: str
    model_v1: str
    model_v2: str
    comparisons: list[ComparisonResult]
    verdict: RegressionVerdict
    evaluators: list[str]
    total_cases: int
    aggregate_stats: dict[str, Any] = Field(default_factory=dict)


class ArenaModelSummary(BaseModel):
    """Aggregate metrics for a single model/prompt in the Arena."""

    model_config = ConfigDict(frozen=True)

    name: str
    model: str
    total_cost: float
    avg_latency_ms: float
    avg_tokens: float
    avg_eval_scores: dict[str, float] = Field(default_factory=dict)
    rank: int = 1
    p_value: float | None = None
    confidence_interval: tuple[float, float] | None = None


class ArenaReport(BaseModel):
    """Leaderboard report across N competing models/prompts."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(default_factory=lambda: f"arena_{uuid.uuid4().hex[:8]}")
    timestamp: str = ""
    variants: list[str]
    models: dict[str, str]
    total_cases: int
    leaderboard: list[ArenaModelSummary]
    comparisons: list[MultiComparisonResult]
