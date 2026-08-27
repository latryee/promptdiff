"""Evaluator Registry & Factory Resolution."""

from __future__ import annotations

from typing import Dict, List, Type
from promptdiff.evaluators.base import BaseEvaluator
from promptdiff.evaluators.cost import CostEvaluator
from promptdiff.evaluators.json_validity import JsonValidityEvaluator
from promptdiff.evaluators.latency import LatencyEvaluator
from promptdiff.evaluators.length_drift import LengthDriftEvaluator
from promptdiff.evaluators.regex_match import RegexMatchEvaluator
from promptdiff.evaluators.similarity import SimilarityEvaluator

EVALUATOR_MAP: Dict[str, Type[BaseEvaluator]] = {
    "json_validity": JsonValidityEvaluator,
    "json": JsonValidityEvaluator,
    "latency": LatencyEvaluator,
    "time": LatencyEvaluator,
    "cost": CostEvaluator,
    "price": CostEvaluator,
    "similarity": SimilarityEvaluator,
    "semantic": SimilarityEvaluator,
    "regex_match": RegexMatchEvaluator,
    "regex": RegexMatchEvaluator,
    "length_drift": LengthDriftEvaluator,
    "length": LengthDriftEvaluator,
}


def get_evaluators(evaluator_names: List[str]) -> List[BaseEvaluator]:
    """Resolve evaluator names into instantiated BaseEvaluator objects.

    Supports comma-separated strings (e.g. 'json_validity,latency,cost').
    """
    instances: List[BaseEvaluator] = []
    seen = set()

    for item in evaluator_names:
        for name in item.split(","):
            clean_name = name.strip().lower()
            if not clean_name or clean_name in seen:
                continue

            if clean_name in EVALUATOR_MAP:
                evaluator_cls = EVALUATOR_MAP[clean_name]
                instances.append(evaluator_cls())
                seen.add(clean_name)

    # Always ensure baseline metrics if nothing specified
    if not instances:
        return [
            JsonValidityEvaluator(),
            LatencyEvaluator(),
            CostEvaluator(),
            SimilarityEvaluator(),
        ]

    return instances
