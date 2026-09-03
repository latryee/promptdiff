"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from promptdiff.core.models import PromptVersion
from promptdiff.production.replay import ShadowTrafficReplayer


@pytest.mark.asyncio
async def test_shadow_replayer(tmp_path: Path) -> None:
    """Test shadow traffic replayer."""
    log_file = tmp_path / "prod.jsonl"
    log_file.write_text('{"query": "Contact user at john.doe@example.com for order #4491"}\n', encoding="utf-8")

    pv = PromptVersion(name="candidate", template="Help: {{query}}")
    replayer = ShadowTrafficReplayer(candidate_prompt=pv, force_mock=True)
    rep = await replayer.replay(str(log_file))
    assert rep.pii_records_sanitized >= 1
    assert rep.total_logs_processed == 1
