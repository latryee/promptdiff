"""Coverage tests for PropertyBasedTester in generators/property_tester.py."""

from __future__ import annotations

import pytest

from promptdiff.core.models import PromptVersion
from promptdiff.generators.property_tester import (
    PropertyBasedTester,
    PropertyInvariant,
    PropertyTestReport,
)


@pytest.mark.asyncio
async def test_property_tester_all_hold() -> None:
    pv = PromptVersion(name="v1", template="Hello {{name}}: {{query}}")
    tester = PropertyBasedTester(
        prompt_version=pv,
        num_iterations=5,
        force_mock=True,
    )
    report = await tester.run_property_tests()

    assert isinstance(report, PropertyTestReport)
    assert report.total_permutations_tested == 5
    assert report.all_invariants_hold is True
    assert report.invariants_violated == 0
    assert len(report.failing_examples) == 0


@pytest.mark.asyncio
async def test_property_tester_custom_invariant_violation() -> None:
    pv = PromptVersion(name="v1", template="Hello {{name}}: {{query}}")
    impossible_inv = PropertyInvariant(
        name="Impossible Condition",
        description="Fails unconditionally",
        check_fn=lambda out, vars: False,
    )
    tester = PropertyBasedTester(
        prompt_version=pv,
        invariants=[impossible_inv],
        num_iterations=3,
        force_mock=True,
    )
    report = await tester.run_property_tests()

    assert report.all_invariants_hold is False
    assert report.invariants_violated == 3
    assert len(report.failing_examples) == 3
