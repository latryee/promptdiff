"""Structured JSON & Contextual Logging Subsystem for promptdiff.

Provides cloud-ready (Datadog, CloudWatch, ELK, Splunk) structured JSON logging
with correlation ID and run ID tracking via Python contextvars.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional

correlation_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("correlation_id", default=None)
run_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("run_id", default=None)


def set_correlation_id(correlation_id: str | None) -> contextvars.Token[Optional[str]]:
    """Set correlation ID in current async / thread context."""
    return correlation_id_ctx.set(correlation_id)


def get_correlation_id() -> str | None:
    """Retrieve current correlation ID."""
    return correlation_id_ctx.get()


def set_run_id(run_id: str | None) -> contextvars.Token[Optional[str]]:
    """Set promptdiff evaluation run_id in current context."""
    return run_id_ctx.set(run_id)


def get_run_id() -> str | None:
    """Retrieve current evaluation run_id."""
    return run_id_ctx.get()


class JSONLogFormatter(logging.Formatter):
    """Structured JSON log formatter outputting machine-parseable log lines."""

    def __init__(self, service_name: str = "promptdiff", **extra_fields: Any):
        super().__init__()
        self.service_name = service_name
        self.extra_fields = extra_fields

    def format(self, record: logging.LogRecord) -> str:
        """Format a standard logging.LogRecord into a single-line JSON document."""
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "source": f"{record.filename}:{record.lineno}",
        }

        # Inject context variables
        cid = correlation_id_ctx.get()
        if cid:
            log_entry["correlation_id"] = cid

        rid = run_id_ctx.get()
        if rid:
            log_entry["run_id"] = rid

        # Record-level overrides
        if hasattr(record, "correlation_id") and record.correlation_id:
            log_entry["correlation_id"] = record.correlation_id
        if hasattr(record, "run_id") and record.run_id:
            log_entry["run_id"] = record.run_id

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Append arbitrary record attributes
        skip_keys = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "correlation_id",
            "run_id",
        }
        for key, value in record.__dict__.items():
            if key not in skip_keys:
                try:
                    json.dumps(value)
                    log_entry[key] = value
                except (TypeError, ValueError):
                    log_entry[key] = str(value)

        log_entry.update(self.extra_fields)
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(
    log_format: str = "text",
    level: str = "INFO",
    logger_name: str | None = None,
    stream: Any = None,
) -> logging.Logger:
    """Configure logger with structured JSON or human-friendly text format.

    Args:
        log_format: 'json' or 'text'.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        logger_name: Name of target logger (None for root logger).
        stream: Output stream (defaults to sys.stderr).

    Returns:
        The configured logger instance.
    """
    target_logger = logging.getLogger(logger_name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    target_logger.setLevel(numeric_level)

    # Remove existing stream handlers to prevent duplicate lines
    for h in list(target_logger.handlers):
        if isinstance(h, logging.StreamHandler):
            target_logger.removeHandler(h)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(numeric_level)

    if log_format.lower() == "json":
        handler.setFormatter(JSONLogFormatter())
    else:
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"))

    target_logger.addHandler(handler)
    return target_logger
