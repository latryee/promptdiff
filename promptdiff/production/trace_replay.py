"""Production Trace & OpenTelemetry / Langfuse Shadow Replayer.

Ingests real-world production LLM traces from OpenTelemetry, OpenInference, or Langfuse,
masks sensitive PII, and replays them as regression test cases across prompt versions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from promptdiff.core.models import DiffReport, TestCase
from promptdiff.core.runner import PromptDiffRunner
from promptdiff.evaluators.registry import get_evaluators
from promptdiff.providers.registry import get_provider


@dataclass
class TraceSpan:
    """Standardized production LLM trace representation."""

    trace_id: str
    span_id: str
    user_query: str
    variables: dict[str, Any] = field(default_factory=dict)
    latency_ms: Optional[float] = None
    model: Optional[str] = None


@dataclass
class TraceIngestionReport:
    """Summary of ingested production traces transformed into evaluation test cases."""

    total_spans_read: int
    valid_test_cases: int
    masked_pii_count: int
    test_cases: list[TestCase] = field(default_factory=list)


class ProductionTraceReplayer:
    """Extracts, sanitizes, and replays production trace distributions through promptdiff."""

    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    PHONE_PATTERN = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")

    def __init__(self, mask_pii: bool = True):
        self.mask_pii = mask_pii

    def _sanitize_text(self, text: str) -> tuple[str, int]:
        """Mask PII (emails and phone numbers) in trace input queries."""
        if not self.mask_pii or not isinstance(text, str):
            return text, 0

        masked_count = 0
        emails = self.EMAIL_PATTERN.findall(text)
        phones = self.PHONE_PATTERN.findall(text)
        masked_count = len(emails) + len(phones)

        clean = self.EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
        clean = self.PHONE_PATTERN.sub("[REDACTED_PHONE]", clean)
        return clean, masked_count

    def ingest_traces(
        self,
        traces_input: Union[str, Path, list[dict[str, Any]]],
        limit: Optional[int] = None,
    ) -> TraceIngestionReport:
        """Parse OTel / Langfuse traces and convert into TestCase instances."""
        raw_items: list[dict[str, Any]] = []

        if isinstance(traces_input, (str, Path)):
            path = Path(traces_input)
            if not path.exists():
                raise FileNotFoundError(f"Trace dataset file not found: {path}")

            text = path.read_text(encoding="utf-8")
            # Support both JSON array and JSONL
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    raw_items = parsed
                elif isinstance(parsed, dict):
                    extracted = parsed.get("traces", parsed.get("data", [parsed]))
                    if isinstance(extracted, list):
                        raw_items = extracted
            except json.JSONDecodeError:
                for line in text.splitlines():
                    if line.strip():
                        raw_items.append(json.loads(line.strip()))
        elif isinstance(traces_input, list):
            raw_items = traces_input

        total_read = len(raw_items)
        if limit and limit > 0:
            raw_items = raw_items[:limit]

        cases: list[TestCase] = []
        total_pii_masked = 0

        for i, item in enumerate(raw_items):
            # Extract user query across common schema formats (Langfuse, OpenInference, OTel)
            query = ""
            if "query" in item:
                query = str(item["query"])
            elif "input" in item:
                inp = item["input"]
                query = str(inp.get("query", inp.get("prompt", str(inp)))) if isinstance(inp, dict) else str(inp)
            elif "attributes" in item:
                # OpenInference OTel span format
                attrs = item["attributes"]
                query = str(attrs.get("llm.input_messages.0.message.content", attrs.get("input.value", "")))
            elif "user_query" in item:
                query = str(item["user_query"])

            if not query:
                continue

            clean_query, pii_count = self._sanitize_text(query)
            total_pii_masked += pii_count

            trace_id = str(item.get("trace_id", item.get("id", f"trace_{i + 1}")))
            vars_dict: dict[str, Any] = item.get("vars", {})
            vars_dict["query"] = clean_query

            cases.append(
                TestCase(
                    id=trace_id,
                    description=f"Replayed production trace ({trace_id})",
                    vars=vars_dict,
                    tags=["production_trace", "shadow_replay"],
                )
            )

        return TraceIngestionReport(
            total_spans_read=total_read,
            valid_test_cases=len(cases),
            masked_pii_count=total_pii_masked,
            test_cases=cases,
        )

    def replay_sync(
        self,
        v1_prompt_text: str,
        v2_prompt_text: str,
        traces_input: Union[str, Path, list[dict[str, Any]]],
        model: str = "gpt-4o",
        mock: bool = True,
        limit: Optional[int] = None,
    ) -> DiffReport:
        """Synchronously replay production traces through PromptDiffRunner."""
        import asyncio

        return asyncio.run(
            self.replay_async(
                v1_prompt_text=v1_prompt_text,
                v2_prompt_text=v2_prompt_text,
                traces_input=traces_input,
                model=model,
                mock=mock,
                limit=limit,
            )
        )

    async def replay_async(
        self,
        v1_prompt_text: str,
        v2_prompt_text: str,
        traces_input: Union[str, Path, list[dict[str, Any]]],
        model: str = "gpt-4o",
        mock: bool = True,
        limit: Optional[int] = None,
    ) -> DiffReport:
        """Asynchronously replay production traces through PromptDiffRunner."""
        from promptdiff.core.models import PromptVersion

        ingestion = self.ingest_traces(traces_input, limit=limit)
        if not ingestion.test_cases:
            raise ValueError("No valid user queries extracted from provided production traces.")

        pv1 = PromptVersion(name="v1", template=v1_prompt_text, model=model)
        pv2 = PromptVersion(name="v2", template=v2_prompt_text, model=model)

        prov1 = get_provider(model_name=model, force_mock=mock)
        prov2 = get_provider(model_name=model, force_mock=mock)
        evals = get_evaluators(["json_validity,latency,cost,similarity"])

        runner = PromptDiffRunner(
            v1_prompt=pv1,
            v2_prompt=pv2,
            provider_v1=prov1,
            provider_v2=prov2,
            evaluators=evals,
        )

        return await runner.run(ingestion.test_cases)


def replay_production_traces(
    v1_prompt: str,
    v2_prompt: str,
    traces: Union[str, Path, list[dict[str, Any]]],
    model: str = "gpt-4o",
    mock: bool = True,
    limit: Optional[int] = None,
) -> DiffReport:
    """Replay real-world production traces from OTel/Langfuse across prompt revisions."""
    replayer = ProductionTraceReplayer()
    return replayer.replay_sync(
        v1_prompt_text=v1_prompt,
        v2_prompt_text=v2_prompt,
        traces_input=traces,
        model=model,
        mock=mock,
        limit=limit,
    )
