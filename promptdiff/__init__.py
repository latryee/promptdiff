"""promptdiff: LLM Prompt & Output Regression Tester CLI.

A developer-first CLI for detecting silent regressions, format breakages,
cost spikes, latency inflation, and output drift across LLM prompt iterations.
"""

__version__ = "0.1.0"
__author__ = "promptdiff team"

from promptdiff.core.models import (
    ComparisonResult,
    DiffReport,
    PromptVersion,
    RegressionVerdict,
    RunResult,
    TestCase,
)

__all__ = [
    "__version__",
    "PromptVersion",
    "TestCase",
    "RunResult",
    "ComparisonResult",
    "RegressionVerdict",
    "DiffReport",
]
