"""Standardized Exit Codes for PromptDiff CLI and CI/CD Automation."""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    """Machine-readable process exit codes for CI/CD pipelines.

    Enables CI workflows to reliably distinguish between:
    - Clean test execution (0)
    - Quality or performance regressions (1)
    - Invalid user configuration or missing files (2)
    - External provider API errors / auth failures (3)
    - Internal framework bugs or unhandled exceptions (4)
    """

    SUCCESS = 0
    REGRESSION_DETECTED = 1
    CONFIGURATION_ERROR = 2
    PROVIDER_ERROR = 3
    INTERNAL_ERROR = 4


# Convenience aliases
EXIT_SUCCESS = ExitCode.SUCCESS
EXIT_REGRESSION = ExitCode.REGRESSION_DETECTED
EXIT_CONFIG_ERROR = ExitCode.CONFIGURATION_ERROR
EXIT_PROVIDER_ERROR = ExitCode.PROVIDER_ERROR
EXIT_INTERNAL_ERROR = ExitCode.INTERNAL_ERROR
