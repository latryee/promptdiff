"""OpenTelemetry & Langfuse / Observability Exporter for promptdiff.

Exports prompt regression test telemetry, token costs, latencies, and evaluator scores
to OpenTelemetry (OTLP JSON) collectors, Langfuse, or Arize Phoenix.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import httpx

from promptdiff.core.models import DiffReport

logger = logging.getLogger("promptdiff.reporters.otel")


def export_to_opentelemetry(
    report: DiffReport,
    endpoint: Optional[str] = None,
    service_name: str = "promptdiff",
    headers: Optional[dict[str, str]] = None,
) -> bool:
    """Export evaluation trace spans to standard OpenTelemetry HTTP collector (OTLP/HTTP JSON)."""
    target_endpoint: str = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "http://localhost:4318/v1/traces"
    auth_header = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")

    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    elif auth_header:
        for pair in auth_header.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                req_headers[k.strip()] = v.strip()

    # Build standard OTLP JSON trace payload
    now_ns = int(time.time() * 1_000_000_000)
    spans = []

    for idx, comp in enumerate(report.comparisons, start=1):
        v1_res = comp.v1_result
        v2_res = comp.v2_result
        tc = comp.test_case

        attributes = [
            {"key": "promptdiff.test_id", "value": {"stringValue": tc.id}},
            {"key": "promptdiff.model_v1", "value": {"stringValue": v1_res.model}},
            {"key": "promptdiff.model_v2", "value": {"stringValue": v2_res.model}},
            {"key": "promptdiff.cost_v1", "value": {"doubleValue": v1_res.cost_usd}},
            {"key": "promptdiff.cost_v2", "value": {"doubleValue": v2_res.cost_usd}},
            {"key": "promptdiff.latency_v1_ms", "value": {"doubleValue": v1_res.latency_ms}},
            {"key": "promptdiff.latency_v2_ms", "value": {"doubleValue": v2_res.latency_ms}},
        ]

        for ev_name, sc in comp.scores.items():
            if isinstance(sc.v2_score, (int, float)):
                attributes.append(
                    {
                        "key": f"promptdiff.score.{ev_name}",
                        "value": {"doubleValue": float(sc.v2_score)},
                    }
                )

        spans.append(
            {
                "traceId": f"{int(time.time()):016x}{idx:016x}",
                "spanId": f"{idx:016x}",
                "name": f"promptdiff/{tc.id}",
                "kind": 1,  # SPAN_KIND_INTERNAL
                "startTimeUnixNano": str(now_ns - int(v2_res.latency_ms * 1_000_000)),
                "endTimeUnixNano": str(now_ns),
                "attributes": attributes,
                "status": {"code": 1 if all(s.passed for s in comp.scores.values()) else 2},
            }
        )

    import promptdiff

    payload = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": service_name}},
                        {"key": "service.version", "value": {"stringValue": promptdiff.__version__}},
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "promptdiff.evaluator"},
                        "spans": spans,
                    }
                ],
            }
        ]
    }

    try:
        with httpx.Client(timeout=8.0) as client:
            resp = client.post(target_endpoint, headers=req_headers, json=payload)
            if resp.status_code in (200, 202):
                logger.info(f"Successfully exported {len(spans)} trace spans to OpenTelemetry: {target_endpoint}")
                return True
            else:
                logger.warning(f"OpenTelemetry collector responded with status {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.warning(f"Could not connect to OpenTelemetry collector: {e}")

    return False


def export_to_langfuse(
    report: DiffReport,
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    host: Optional[str] = None,
) -> bool:
    """Export evaluation results to Langfuse REST API."""
    pk = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    api_host: str = host or os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com"

    if not pk or not sk:
        logger.debug("Langfuse public/secret keys not provided.")
        return False

    url = f"{api_host.rstrip('/')}/api/public/events"
    events = []

    for comp in report.comparisons:
        events.append(
            {
                "name": f"promptdiff_regression_{comp.test_case.id}",
                "metadata": {
                    "v1_model": comp.v1_result.model,
                    "v2_model": comp.v2_result.model,
                    "v1_cost": comp.v1_result.cost_usd,
                    "v2_cost": comp.v2_result.cost_usd,
                    "v1_latency": comp.v1_result.latency_ms,
                    "v2_latency": comp.v2_result.latency_ms,
                    "passed": all(s.passed for s in comp.scores.values()),
                },
            }
        )

    try:
        with httpx.Client(timeout=8.0, auth=(pk, sk)) as client:
            resp = client.post(url, json={"events": events})
            if resp.status_code in (200, 201):
                logger.info(f"Successfully exported {len(events)} events to Langfuse: {api_host}")
                return True
    except Exception as e:
        logger.warning(f"Could not export to Langfuse: {e}")

    return False
