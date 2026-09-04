import io
import json
import logging

from promptdiff.core.logging import (
    JSONLogFormatter,
    get_correlation_id,
    get_run_id,
    set_correlation_id,
    set_run_id,
    setup_logging,
)


def test_json_log_formatter_basic() -> None:
    formatter = JSONLogFormatter(service_name="test-service")
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Hello %s",
        args=("world",),
        exc_info=None,
    )

    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert data["message"] == "Hello world"
    assert data["level"] == "INFO"
    assert data["logger"] == "test_logger"
    assert data["service"] == "test-service"
    assert "timestamp" in data
    assert "source" in data


def test_json_log_formatter_with_context() -> None:
    formatter = JSONLogFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.WARNING,
        pathname=__file__,
        lineno=25,
        msg="Warning event",
        args=(),
        exc_info=None,
    )

    token_cid = set_correlation_id("corr-xyz-123")
    token_rid = set_run_id("run-abc-789")
    try:
        assert get_correlation_id() == "corr-xyz-123"
        assert get_run_id() == "run-abc-789"
        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert data["correlation_id"] == "corr-xyz-123"
        assert data["run_id"] == "run-abc-789"
    finally:
        from promptdiff.core.logging import correlation_id_ctx, run_id_ctx

        correlation_id_ctx.reset(token_cid)
        run_id_ctx.reset(token_rid)


def test_json_log_formatter_extra_fields() -> None:
    formatter = JSONLogFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.ERROR,
        pathname=__file__,
        lineno=40,
        msg="Error event",
        args=(),
        exc_info=None,
    )
    record.__dict__["custom_metric"] = 42.5
    record.__dict__["unserializable"] = object()

    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert data["custom_metric"] == 42.5
    assert "unserializable" in data


def test_setup_logging_json() -> None:
    stream = io.StringIO()
    logger = setup_logging(log_format="json", level="DEBUG", logger_name="json_test", stream=stream)
    logger.info("JSON stream message")

    stream.seek(0)
    line = stream.readline().strip()
    data = json.loads(line)
    assert data["message"] == "JSON stream message"
    assert data["level"] == "INFO"
    assert data["logger"] == "json_test"


def test_setup_logging_text() -> None:
    stream = io.StringIO()
    logger = setup_logging(log_format="text", level="INFO", logger_name="text_test", stream=stream)
    logger.info("Text stream message")

    stream.seek(0)
    line = stream.readline().strip()
    assert "Text stream message" in line
    assert "[INFO]" in line
    assert "[text_test]" in line
