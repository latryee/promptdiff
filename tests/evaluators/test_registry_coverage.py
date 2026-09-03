"""Coverage boost tests for evaluators/registry.py — get_evaluators factory."""

from __future__ import annotations

import pytest

from promptdiff.evaluators.base import BaseEvaluator
from promptdiff.evaluators.registry import EVALUATOR_MAP, get_evaluators


def test_get_evaluators_single() -> None:
    evals = get_evaluators(["json_validity"])
    assert len(evals) == 1
    assert isinstance(evals[0], BaseEvaluator)
    assert evals[0].name.lower().replace("_", "").replace(" ", "") != ""


def test_get_evaluators_csv_string() -> None:
    evals = get_evaluators(["json_validity,latency,cost"])
    assert len(evals) == 3


def test_get_evaluators_multiple_names() -> None:
    evals = get_evaluators(["similarity", "regex_match", "length_drift"])
    assert len(evals) == 3


def test_get_evaluators_all_aliases_resolve() -> None:
    """Every key in EVALUATOR_MAP should resolve to a valid evaluator."""
    for alias in EVALUATOR_MAP:
        evals = get_evaluators([alias])
        assert len(evals) == 1
        assert isinstance(evals[0], BaseEvaluator)


def test_get_evaluators_unknown_raises() -> None:
    with pytest.raises((ValueError, KeyError)):
        get_evaluators(["nonexistent_evaluator_xyz"])


def test_get_evaluators_empty_list() -> None:
    evals = get_evaluators([])
    assert len(evals) == 4
