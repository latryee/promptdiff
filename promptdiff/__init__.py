"""promptdiff: Enterprise-grade LLM Prompt & Model Regression Testing Framework.

A developer-first CLI for detecting silent regressions, format breakages,
cost spikes, latency inflation, and output drift across LLM prompt iterations.
"""

__version__ = "2.0.0"
__author__ = "promptdiff team"

from promptdiff.core.models import (
    ArenaReport,
    ComparisonResult,
    DiffReport,
    MultiComparisonResult,
    PromptVersion,
    RegressionVerdict,
    RunResult,
    TestCase,
)

__all__ = [
    "ArenaReport",
    "ComparisonResult",
    "DiffReport",
    "MultiComparisonResult",
    "PromptVersion",
    "RegressionVerdict",
    "RunResult",
    "TestCase",
    "__version__",
]
