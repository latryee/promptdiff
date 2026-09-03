"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import pytest

from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.production.edge_quant import EdgeQuantizationBenchmark


@pytest.mark.asyncio
async def test_edge_quantization_benchmark() -> None:
    """Test local model quantization degradation benchmark."""
    pv = PromptVersion(name="edge_p", template="Explain: {{query}}")
    bench = EdgeQuantizationBenchmark(
        prompt_version=pv, test_cases=[TestCase(id="1", vars={"query": "AI"})], force_mock=True
    )
    report = await bench.benchmark_quant_levels()
    assert len(report.levels) == 5
    assert any(lvl.quant_level.startswith("Q4_K_M") for lvl in report.levels)
    assert "Q4_K_M" in report.optimal_edge_quant
