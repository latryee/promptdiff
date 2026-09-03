"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import pytest

from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.optimizer.exemplars import DynamicExemplarSelector, ExemplarItem
from promptdiff.optimizer.mmr_selector import Exemplar, MMRExemplarSelector


@pytest.mark.asyncio
async def test_exemplars_selector() -> None:
    """Test dynamic few-shot vector indexer."""
    exs = [
        ExemplarItem(input_text="How to cancel?", output_text="Go to settings -> cancel"),
        ExemplarItem(input_text="How to change plan?", output_text="Go to billing -> change"),
    ]
    selector = DynamicExemplarSelector(golden_exemplars=exs, top_k=1)
    retrieved = selector.retrieve_exemplars("I want to cancel my account")
    assert len(retrieved) == 1
    assert "cancel" in retrieved[0].output_text

    rep = await selector.benchmark(
        base_prompt=PromptVersion(name="b", template="Help: {{query}}"),
        test_cases=[TestCase(id="1", vars={"query": "cancel"})],
        force_mock=True,
    )
    assert rep.dynamic_judge_score >= rep.static_judge_score


def test_mmr_exemplar_selector() -> None:
    """Test Maximal Marginal Relevance dynamic exemplar selection."""
    pool = [
        Exemplar(id="1", input_text="How do I reset password?", output_text="Go to settings."),
        Exemplar(id="2", input_text="Password reset instructions", output_text="Click forgot password."),
        Exemplar(id="3", input_text="Where can I see invoices?", output_text="Billing dashboard."),
    ]
    selector = MMRExemplarSelector(diversity_lambda=0.7)
    res = selector.select(query="Reset my user password", pool=pool, top_k=2)
    assert len(res.selected_exemplars) == 2
