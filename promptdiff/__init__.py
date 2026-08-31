"""promptdiff: Enterprise-grade LLM Prompt & Model Regression Testing Framework."""

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
    cache_sim,
    compare,
    export_bundle,
    fuzz,
    history,
    mutate,
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
    "fuzz",
    "cache_sim",
    "mutate",
    "history",
    "export_bundle",
    "__version__",
]
