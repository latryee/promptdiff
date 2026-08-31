"""Property-Based Invariant Testing Engine for promptdiff (promptdiff property-test)."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from promptdiff.core.models import PromptVersion
from promptdiff.providers.registry import get_provider

logger = logging.getLogger("promptdiff.generators.property_tester")


@dataclass
class PropertyInvariant:
    """A logical/mathematical rule that must hold True for all LLM outputs."""

    name: str
    description: str
    check_fn: Callable[[str, dict[str, Any]], bool]


@dataclass
class PropertyTestReport:
    """Outcome of fuzzing invariant properties."""

    prompt_name: str
    total_permutations_tested: int
    invariants_passed: int
    invariants_violated: int
    all_invariants_hold: bool
    failing_examples: list[dict[str, Any]] = field(default_factory=list)


class PropertyBasedTester:
    """Fuzzes dynamic prompt variable spaces to detect invariant violations."""

    def __init__(
        self,
        prompt_version: PromptVersion,
        invariants: Optional[list[PropertyInvariant]] = None,
        model_name: str = "gpt-4o",
        num_iterations: int = 15,
        force_mock: bool = False,
    ):
        self.prompt_version = prompt_version
        self.invariants = invariants or [
            PropertyInvariant(
                name="Non-Empty Response",
                description="Model output must never be empty or whitespace only",
                check_fn=lambda out, vars: len(out.strip()) > 0,
            ),
            PropertyInvariant(
                name="No Forbidden Disclaimers",
                description="Model output must not complain 'as an AI language model'",
                check_fn=lambda out, vars: "as an ai language model" not in out.lower(),
            ),
            PropertyInvariant(
                name="Output Length Upper Bound",
                description="Model output must stay under 4000 characters",
                check_fn=lambda out, vars: len(out) < 4000,
            ),
        ]
        self.model_name = model_name
        self.num_iterations = num_iterations
        self.force_mock = force_mock

    def _generate_random_variables(self) -> dict[str, Any]:
        """Generate high-entropy randomized input variables."""
        names = ["Alice", "Bob", "Charlie", "Zaphod", "X_Æ_A-12"]
        queries = [
            "What is my balance?",
            "Can I get a refund for order #0?",
            "Special characters: !@#$%^&*()_+",
            "Very long text " * 50,
            "1234567890",
        ]
        return {
            "name": random.choice(names),
            "query": random.choice(queries),
            "amount": random.randint(0, 100000),
            "id": f"id_{random.randint(100, 9999)}",
        }

    async def run_property_tests(self) -> PropertyTestReport:
        """Run generative invariant tests."""
        provider = get_provider(model_name=self.model_name, force_mock=self.force_mock)
        failing: list[dict[str, Any]] = []

        for i in range(self.num_iterations):
            rand_vars = self._generate_random_variables()
            rendered = self.prompt_version.render(rand_vars)

            if self.force_mock:
                output = f"Processed request for {rand_vars.get('name')}: {rand_vars.get('query')}"
            else:
                try:
                    res = await provider.generate(prompt=rendered, max_tokens=256)
                    output = res.output
                except Exception as e:
                    output = f"Error: {e}"

            # Check invariants
            for inv in self.invariants:
                if not inv.check_fn(output, rand_vars):
                    failing.append({
                        "iteration": i + 1,
                        "invariant": inv.name,
                        "input_vars": rand_vars,
                        "violating_output": output[:150],
                    })

        violations = len(failing)
        passed = (self.num_iterations * len(self.invariants)) - violations

        return PropertyTestReport(
            prompt_name=self.prompt_version.name,
            total_permutations_tested=self.num_iterations,
            invariants_passed=passed,
            invariants_violated=violations,
            all_invariants_hold=violations == 0,
            failing_examples=failing,
        )
