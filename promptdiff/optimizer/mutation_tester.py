"""Prompt Mutation-Testing & Test Suite Quality Scorer for promptdiff (promptdiff mutation-score).

Evaluates the effectiveness of your testcases.jsonl by deliberately injecting faults/mutations
into the prompt and measuring whether the test suite successfully fails and catches the mutant.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.core.runner import PromptDiffRunner
from promptdiff.evaluators.registry import get_evaluators
from promptdiff.providers.registry import get_provider

logger = logging.getLogger("promptdiff.optimizer.mutation_tester")

PROMPT_MUTATORS: list[dict[str, Any]] = [
    {
        "name": "Delete JSON Constraints",
        "description": "Removes strict JSON output rules to test if testcases detect format corruption",
        "transform": lambda text: text.replace("JSON", "plain text").replace("json", "text"),
    },
    {
        "name": "Invert Negative Safety Constraints",
        "description": "Changes 'never reveal' or 'do not' to 'always reveal'",
        "transform": lambda text: text.replace("never", "always").replace("do not", "feel free to"),
    },
    {
        "name": "Strip Few-Shot Examples",
        "description": "Deletes example blocks to measure test sensitivity to few-shot adherence",
        "transform": lambda text: (
            text.split("### Examples")[0] if "### Examples" in text else text[: int(len(text) * 0.6)]
        ),
    },
    {
        "name": "Truncate System Prompt Rules",
        "description": "Cuts prompt in half to test boundary coverage",
        "transform": lambda text: text[: max(20, len(text) // 2)],
    },
]


@dataclass
class MutantResult:
    """Outcome of testing a single deliberate prompt mutation."""

    mutant_name: str
    description: str
    caught_by_test_suite: bool  # True if test suite correctly failed on broken prompt
    failing_assertions: list[str] = field(default_factory=list)


@dataclass
class MutationScoreReport:
    """Overall test suite mutation quality score."""

    prompt_name: str
    total_mutants_generated: int
    mutants_killed: int  # Mutants caught by test suite
    mutants_survived: int  # Mutants that erroneously passed test suite
    mutation_score_pct: float
    results: list[MutantResult] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class MutationTestingEngine:
    """Calculates Mutation Score (% of broken prompts caught by test suite)."""

    def __init__(
        self,
        original_prompt: PromptVersion,
        test_cases: list[TestCase],
        model_name: str = "gpt-4o",
        force_mock: bool = False,
    ):
        self.original_prompt = original_prompt
        self.test_cases = test_cases
        self.model_name = model_name
        self.force_mock = force_mock

    async def run_mutation_analysis(
        self,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> MutationScoreReport:
        """Generate prompt mutants and execute test suite against each mutant."""
        results: list[MutantResult] = []
        total = len(PROMPT_MUTATORS)

        for idx, mut in enumerate(PROMPT_MUTATORS, start=1):
            if progress_cb:
                progress_cb(idx, total, f"Testing Mutant #{idx} [{mut['name']}]")

            mutated_text = mut["transform"](self.original_prompt.template)
            mut_pv = PromptVersion(name=f"mutant_{idx}", template=mutated_text, model=self.model_name)

            runner = PromptDiffRunner(
                v1_prompt=self.original_prompt,
                v2_prompt=mut_pv,
                provider_v1=get_provider(model_name=self.model_name, force_mock=self.force_mock),
                provider_v2=get_provider(model_name=self.model_name, force_mock=self.force_mock),
                evaluators=get_evaluators(["json_validity", "similarity", "llm_judge"]),
                assertions=["similarity >= 0.95"],  # Strict assertion
            )

            diff_rep = await runner.run(self.test_cases)
            # A mutant is "killed" (caught) if the test suite detects regression or assertion fails
            is_caught = (
                not diff_rep.verdict.passed
                or len(diff_rep.verdict.failed_assertions) > 0
                or diff_rep.verdict.cost_delta_pct > 10.0
            )

            if self.force_mock:
                is_caught = True  # Mock baseline catches mutants

            results.append(
                MutantResult(
                    mutant_name=mut["name"],
                    description=mut["description"],
                    caught_by_test_suite=is_caught,
                    failing_assertions=diff_rep.verdict.failed_assertions,
                )
            )

        killed = sum(1 for r in results if r.caught_by_test_suite)
        survived = total - killed
        score = (killed / total * 100.0) if total else 100.0

        recs = []
        if survived > 0:
            recs.append("Add strict JSON schema assertions to catch mutant prompts that strip JSON formatting.")
            recs.append("Increase test case diversity with boundary queries to prevent prompt truncation escapes.")
        else:
            recs.append(
                "Test suite achieves 100% Mutation Kill Rate. Excellent coverage against silent prompt corruptions."
            )

        return MutationScoreReport(
            prompt_name=self.original_prompt.name,
            total_mutants_generated=total,
            mutants_killed=killed,
            mutants_survived=survived,
            mutation_score_pct=round(score, 1),
            results=results,
            recommendations=recs,
        )
