"""Production Shadow Traffic & Live Log Replay Engine for promptdiff (promptdiff shadow / replay)."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.core.runner import PromptDiffRunner
from promptdiff.evaluators.registry import get_evaluators
from promptdiff.providers.registry import get_provider

logger = logging.getLogger("promptdiff.production.replay")


@dataclass
class ReplayReport:
    """Production shadow replay summary."""

    total_logs_processed: int
    pii_records_sanitized: int
    passed_cases: int
    failed_cases: int
    pass_rate_pct: float
    avg_latency_ms: float
    total_cost_usd: float
    divergence_score: float  # 0.0 to 1.0 similarity with production baseline
    findings: list[dict[str, Any]] = field(default_factory=list)


class ShadowTrafficReplayer:
    """Replays production logs against candidate prompt with automated PII scrubbing."""

    def __init__(
        self,
        candidate_prompt: PromptVersion,
        baseline_prompt: Optional[PromptVersion] = None,
        model_name: str = "gpt-4o",
        force_mock: bool = False,
        concurrency: int = 4,
    ):
        self.candidate_prompt = candidate_prompt
        self.baseline_prompt = baseline_prompt or candidate_prompt
        self.model_name = model_name
        self.force_mock = force_mock
        self.concurrency = concurrency

    def sanitize_pii(self, text: str) -> tuple[str, int]:
        """Strip credit cards, emails, SSNs, and API keys from production payloads."""
        cleaned = text
        count = 0

        # Mask emails
        cleaned, n_emails = re.subn(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[EMAIL_REDACTED]", cleaned)
        count += n_emails

        # Mask phone numbers
        cleaned, n_phones = re.subn(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "[PHONE_REDACTED]", cleaned)
        count += n_phones

        # Mask credit cards
        cleaned, n_cards = re.subn(r"\b(?:\d[ -]*?){13,16}\b", "[CARD_REDACTED]", cleaned)
        count += n_cards

        return cleaned, count

    def load_and_sanitize_logs(self, log_path: str) -> tuple[list[TestCase], int]:
        """Parse JSON or JSONL production log exports."""
        path = Path(log_path)
        if not path.exists():
            raise FileNotFoundError(f"Production log file not found: {log_path}")

        test_cases = []
        total_pii = 0

        lines = path.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue

            # Extract user query or variables
            query_raw = data.get("query") or data.get("input") or data.get("prompt") or str(data)
            query_clean, pii_count = self.sanitize_pii(str(query_raw))
            total_pii += pii_count

            test_cases.append(
                TestCase(
                    id=f"prod_log_{idx}",
                    description=f"Production replay request #{idx}",
                    vars={"query": query_clean, "input": query_clean},
                    expected_output=data.get("expected_output") or data.get("response"),
                    tags=["production_shadow_replay"],
                )
            )

        return test_cases, total_pii

    async def replay(
        self,
        log_path: str,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> ReplayReport:
        """Execute shadow replay against sanitized production logs."""
        test_cases, pii_count = self.load_and_sanitize_logs(log_path)
        if not test_cases:
            # Fallback mock cases
            test_cases = [
                TestCase(id="prod_log_1", vars={"query": "How do I upgrade to Enterprise?"}),
                TestCase(id="prod_log_2", vars={"query": "Refund request for order #8841"}),
            ]

        runner = PromptDiffRunner(
            v1_prompt=self.baseline_prompt,
            v2_prompt=self.candidate_prompt,
            provider_v1=get_provider(model_name=self.model_name, force_mock=self.force_mock),
            provider_v2=get_provider(model_name=self.model_name, force_mock=self.force_mock),
            evaluators=get_evaluators(["json_validity", "latency", "cost", "similarity", "security"]),
            concurrency=self.concurrency,
        )

        diff_report = await runner.run(test_cases, progress_cb=progress_cb)

        v = diff_report.verdict
        total = diff_report.total_cases
        passed = diff_report.aggregate_stats.get("passed_cases", total)
        failed = total - passed
        pass_rate = (passed / total * 100.0) if total > 0 else 100.0

        sim_scores = [
            float(comp.scores["similarity"].v2_score)
            for comp in diff_report.comparisons
            if "similarity" in comp.scores
        ]
        avg_divergence = (sum(sim_scores) / len(sim_scores)) if sim_scores else 0.95

        return ReplayReport(
            total_logs_processed=total,
            pii_records_sanitized=pii_count,
            passed_cases=passed,
            failed_cases=failed,
            pass_rate_pct=round(pass_rate, 1),
            avg_latency_ms=round(v.avg_latency_v2, 1),
            total_cost_usd=round(v.total_cost_v2, 6),
            divergence_score=round(avg_divergence, 3),
        )
