"""Structured exception hierarchy for promptdiff.

Provides distinct exception classes for caching, provider execution,
evaluators, and configuration to eliminate unhandled bare exceptions.
"""

from __future__ import annotations


class PromptDiffError(Exception):
    """Base exception for all promptdiff errors."""


class CacheError(PromptDiffError):
    """Raised or logged when persistent cache operations fail."""


class CacheReadError(CacheError):
    """Raised when reading from cache fails."""


class CacheWriteError(CacheError):
    """Raised when writing to cache fails."""


class RunnerError(PromptDiffError):
    """Base exception for test runner execution errors."""


class ProviderExecutionError(RunnerError):
    """Raised when LLM provider invocation fails during test execution."""


class EvaluatorExecutionError(RunnerError):
    """Raised when an evaluator fails during candidate comparison."""


class DatasetLoadError(PromptDiffError):
    """Raised when test cases dataset fails to parse or load."""


class ConfigurationError(PromptDiffError):
    """Raised when invalid configuration or parameters are provided."""
