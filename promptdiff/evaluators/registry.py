"""Evaluator Registry & Factory Resolution for promptdiff v3.0."""

from __future__ import annotations

from promptdiff.evaluators.answer_relevance import AnswerRelevanceEvaluator
from promptdiff.evaluators.base import BaseEvaluator
from promptdiff.evaluators.cascaded_judge import CascadedLLMJudgeEvaluator
from promptdiff.evaluators.citation import CitationEvaluator
from promptdiff.evaluators.code_sandbox import SafeCodeSandboxEvaluator
from promptdiff.evaluators.cost import CostEvaluator
from promptdiff.evaluators.council import CouncilOfJudgesEvaluator
from promptdiff.evaluators.debate import MultiAgentDebateEvaluator
from promptdiff.evaluators.exact_match import ExactMatchEvaluator
from promptdiff.evaluators.fact_graph import FactGraphEvaluator
from promptdiff.evaluators.fairness import FairnessEvaluator
from promptdiff.evaluators.faithfulness import FaithfulnessEvaluator
from promptdiff.evaluators.hallucination_graph import TokenAttributionEvaluator
from promptdiff.evaluators.json_validity import JsonValidityEvaluator
from promptdiff.evaluators.latency import LatencyEvaluator
from promptdiff.evaluators.length_drift import LengthDriftEvaluator
from promptdiff.evaluators.llm_judge import LLMJudgeEvaluator
from promptdiff.evaluators.multilingual import MultilingualConsistencyEvaluator
from promptdiff.evaluators.regex_match import RegexMatchEvaluator
from promptdiff.evaluators.schema_breaking import SchemaBreakingChangeEvaluator
from promptdiff.evaluators.schema_repair import SchemaRepairEvaluator
from promptdiff.evaluators.security import SecurityEvaluator
from promptdiff.evaluators.similarity import SimilarityEvaluator
from promptdiff.evaluators.trajectory import TrajectoryEvaluator
from promptdiff.evaluators.vision import VisionDiffEvaluator

EVALUATOR_MAP: dict[str, type[BaseEvaluator]] = {
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
    "llm_judge": LLMJudgeEvaluator,
    "judge": LLMJudgeEvaluator,
    "llm_as_a_judge": LLMJudgeEvaluator,
    "cascaded_judge": CascadedLLMJudgeEvaluator,
    "cascade_judge": CascadedLLMJudgeEvaluator,
    "cost_judge": CascadedLLMJudgeEvaluator,
    "faithfulness": FaithfulnessEvaluator,
    "rag_faithfulness": FaithfulnessEvaluator,
    "groundedness": FaithfulnessEvaluator,
    "answer_relevance": AnswerRelevanceEvaluator,
    "relevance": AnswerRelevanceEvaluator,
    "query_relevance": AnswerRelevanceEvaluator,
    "security": SecurityEvaluator,
    "guardrails": SecurityEvaluator,
    "safety": SecurityEvaluator,
    "trajectory": TrajectoryEvaluator,
    "agent_trajectory": TrajectoryEvaluator,
    "tool_use": TrajectoryEvaluator,
    "fairness": FairnessEvaluator,
    "bias": FairnessEvaluator,
    "citation": CitationEvaluator,
    "cite": CitationEvaluator,
    "schema_repair": SchemaRepairEvaluator,
    "schema_breaking": SchemaBreakingChangeEvaluator,
    "breaking_changes": SchemaBreakingChangeEvaluator,
    "schema_diff": SchemaBreakingChangeEvaluator,
    "vision": VisionDiffEvaluator,
    "council": CouncilOfJudgesEvaluator,
    "swarm_judge": CouncilOfJudgesEvaluator,
    "council_judge": CouncilOfJudgesEvaluator,
    "token_attribution": TokenAttributionEvaluator,
    "rag_grounding": TokenAttributionEvaluator,
    "hallucination_graph": TokenAttributionEvaluator,
    "debate_judge": MultiAgentDebateEvaluator,
    "debate": MultiAgentDebateEvaluator,
    "fact_graph": FactGraphEvaluator,
    "code_sandbox": SafeCodeSandboxEvaluator,
    "sandbox": SafeCodeSandboxEvaluator,
    "multilingual": MultilingualConsistencyEvaluator,
    "exact_match": ExactMatchEvaluator,
    "exact": ExactMatchEvaluator,
}


def register_evaluator(name: str, evaluator_cls: type[BaseEvaluator]) -> None:
    """Register a custom evaluator class under one or more alias names."""
    clean = name.strip().lower()
    EVALUATOR_MAP[clean] = evaluator_cls


def get_evaluators(evaluator_names: list[str]) -> list[BaseEvaluator]:
    """Resolve evaluator names into instantiated BaseEvaluator objects.

    Supports comma-separated strings (e.g. 'json_validity,latency,cost,similarity,exact_match'),
    built-in aliases, custom registered evaluators, and dynamic 'module.path:ClassName' references.
    """
    instances: list[BaseEvaluator] = []
    seen = set()

    for item in evaluator_names:
        for name in item.split(","):
            raw_clean = name.strip()
            clean_name = raw_clean.lower()
            if not clean_name or clean_name in seen:
                continue

            # 1. Direct registry lookup
            if clean_name in EVALUATOR_MAP:
                evaluator_cls = EVALUATOR_MAP[clean_name]
                instances.append(evaluator_cls())
                seen.add(clean_name)
            # 2. Dynamic module:Class import
            elif ":" in raw_clean:
                import importlib

                mod_path, cls_name = raw_clean.split(":", 1)
                try:
                    mod = importlib.import_module(mod_path.strip())
                    custom_cls = getattr(mod, cls_name.strip())
                    if not issubclass(custom_cls, BaseEvaluator):
                        raise TypeError(f"Custom class {cls_name} must subclass BaseEvaluator")
                    instances.append(custom_cls())
                    seen.add(clean_name)
                except Exception as import_err:
                    raise ValueError(f"Failed to load custom evaluator '{raw_clean}': {import_err}") from import_err
            else:
                import difflib

                matches = difflib.get_close_matches(clean_name, list(EVALUATOR_MAP.keys()), n=3, cutoff=0.5)
                suggestion = f" Did you mean: {', '.join(matches)}?" if matches else ""
                raise ValueError(
                    f"Unknown evaluator '{clean_name}'.{suggestion} "
                    f"Supported evaluators: {', '.join(sorted(set(EVALUATOR_MAP.keys())))}"
                )

    if not instances:
        return [
            JsonValidityEvaluator(),
            LatencyEvaluator(),
            CostEvaluator(),
            SimilarityEvaluator(),
        ]

    return instances
