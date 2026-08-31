"""promptdiff: Enterprise-grade LLM Prompt & Model Regression Testing Framework.

A developer-first framework and CLI for detecting silent regressions, format breakages,
cost spikes, latency inflation, output drift, and security leaks across LLM prompt iterations.
"""

__version__ = "3.0.0"
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
from promptdiff.sdk import (
    async_compare,
    compare,
    optimize,
    shrink,
    tune,
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
    "async_compare",
    "compare",
    "optimize",
    "tune",
    "shrink",
    "__version__",
]
