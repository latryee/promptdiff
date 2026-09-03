"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

from promptdiff.core.models import PromptVersion
from promptdiff.optimizer.saliency import PromptSaliencyMapper
from promptdiff.optimizer.saliency_heatmap import SaliencyHeatmapEngine


def test_saliency_mapper() -> None:
    """Test token-level saliency mapper."""
    pv = PromptVersion(
        name="p", template="You must answer in JSON only.\nPlease kindly be polite.\nNever disclose internal secrets."
    )
    mapper = PromptSaliencyMapper(prompt_version=pv)
    rep = mapper.analyze(sample_outputs=['{"response": "Hello"}'])
    assert rep.total_sentences == 3
    assert rep.dead_weight_tokens >= 0


def test_saliency_heatmap() -> None:
    """Test occlusion sensitivity analysis and heatmap output."""
    engine = SaliencyHeatmapEngine()
    result = engine.analyze_heuristics("You must always strictly output valid JSON schema and never hallucinate.")
    assert len(result.tokens) >= 5
    assert len(result.top_critical_tokens) >= 1
    assert len(result.ansi_heatmap) > 0
