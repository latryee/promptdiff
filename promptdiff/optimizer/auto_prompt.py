"""Auto-Prompt Optimizer (DSPy-style Reflection & Meta-Prompting).

Iteratively identifies failure modes from test cases and LLM Judge criticism,
feeding error feedback into a Meta-LLM to generate refined, resilient prompts.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from promptdiff.core.models import DiffReport, PromptVersion, TestCase
from promptdiff.core.runner import PromptDiffRunner
from promptdiff.evaluators.base import BaseEvaluator
from promptdiff.evaluators.registry import get_evaluators
from promptdiff.providers.base import BaseLLMProvider
from promptdiff.providers.registry import get_provider

logger = logging.getLogger("promptdiff.optimizer")

META_OPTIMIZER_PROMPT = """
You are an expert Meta-Prompt Optimization Engine (DSPy / MIPRO style prompt engineer).
Your goal is to rewrite, refine, and optimize an LLM prompt template that failed specific test cases.

--- CURRENT PROMPT TEMPLATE ---
{current_prompt}

--- TARGET TASK DESCRIPTION ---
{task_description}

--- FAILING TEST CASES & EVALUATOR CRITICISM ---
{failure_summary}

--- OPTIMIZATION INSTRUCTIONS ---
1. Analyze the root causes of the failures (e.g. format disobedience, hallucination, verbosity, ambiguous instructions, missed edge cases).
2. Rewrite the PROMPT TEMPLATE to solve all identified failure modes while preserving original variable placeholders like {{var_name}} or {var_name}.
3. Keep the prompt concise, robust, and unambiguous. Include explicit constraints or formatting directives if needed.
4. Return ONLY the new optimized prompt template enclosed within ```prompt ... ``` blocks.

```prompt
<YOUR_OPTIMIZED_PROMPT_HERE>
```
"""


@dataclass
class OptimizationResult:
    """Outcome of an auto-prompt optimization run."""

    original_prompt: str
    optimized_prompt: str
    initial_pass_rate: float
    final_pass_rate: float
    iterations: int
    failed_cases_addressed: int
    initial_report: DiffReport
    final_report: DiffReport
    history: list[dict[str, Any]] = field(default_factory=list)


class PromptOptimizer:
    """Reflective DSPy-style Auto-Prompt Optimizer."""

    def __init__(
        self,
        prompt_version: PromptVersion,
        test_cases: list[TestCase],
        evaluators: list[BaseEvaluator] | None = None,
        provider: BaseLLMProvider | None = None,
        meta_provider: BaseLLMProvider | None = None,
        model_name: str = "gpt-4o",
        meta_model_name: str = "gpt-4o",
        max_iterations: int = 3,
        force_mock: bool = False,
    ):
        self.prompt_version = prompt_version
        self.test_cases = test_cases
        self.evaluators = evaluators or get_evaluators(["json_validity", "latency", "cost", "similarity", "llm_judge"])
        self.provider = provider or get_provider(model_name=model_name, force_mock=force_mock)
        self.meta_provider = meta_provider or get_provider(model_name=meta_model_name, force_mock=force_mock)
        self.max_iterations = max_iterations
        self.force_mock = force_mock

    def _extract_failures(self, report: DiffReport) -> list[dict[str, Any]]:
        """Identify failed or low-scoring test cases and collect judge/evaluator reasoning."""
        failures = []
        for comp in report.comparisons:
            scores = comp.scores
            all_passed = all(s.passed for s in scores.values())
            # Check for failed assertions, low judge score, or errors
            if not all_passed or comp.v2_result.error:
                criticisms = []
                for ev_name, score_obj in scores.items():
                    if not score_obj.passed:
                        reason = score_obj.details.get("reasoning", score_obj.message)
                        criticisms.append(f"[{ev_name.upper()}]: {reason}")

                failures.append(
                    {
                        "test_id": comp.test_case.id,
                        "description": comp.test_case.description,
                        "variables": comp.test_case.vars,
                        "actual_output": comp.v2_result.output[:300],
                        "expected_output": comp.test_case.expected_output,
                        "criticisms": criticisms
                        or [f"Output did not satisfy requirements ({comp.v2_result.error or 'Failed metric'})"],
                    }
                )
        return failures

    def _format_failures_for_meta_prompt(self, failures: list[dict[str, Any]]) -> str:
        """Format failure diagnostics into markdown for Meta-LLM reflection."""
        lines = []
        for idx, f in enumerate(failures[:6], start=1):  # Cap to top 6 failures for prompt budget
            lines.append(f"Failure #{idx} (Test ID: {f['test_id']}):")
            lines.append(f"  Inputs: {json.dumps(f['variables'], ensure_ascii=False)}")
            lines.append(f"  Actual Output: {f['actual_output']}")
            if f.get("expected_output"):
                lines.append(f"  Expected Output: {f['expected_output']}")
            lines.append("  Judge / Evaluator Feedback:")
            for crit in f["criticisms"]:
                lines.append(f"    - {crit}")
            lines.append("")
        return "\n".join(lines)

    def _parse_meta_output(self, meta_text: str, fallback: str) -> str:
        """Extract prompt template from ```prompt ... ``` blocks."""
        match = re.search(r"```(?:prompt)?\s*\n(.*?)\n```", meta_text, re.DOTALL | re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate

        # Fallback to general code block if ```prompt block not explicitly used
        match_code = re.search(r"```\s*\n(.*?)\n```", meta_text, re.DOTALL)
        if match_code:
            candidate = match_code.group(1).strip()
            if candidate:
                return candidate

        return fallback if not meta_text.strip() else meta_text.strip()

    async def optimize(
        self,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> OptimizationResult:
        """Run iterative DSPy-style reflection and prompt optimization loop."""
        current_template = self.prompt_version.template
        best_template = current_template

        # Baseline evaluation (v1 = current, v2 = current)
        v1_p = PromptVersion(
            name="baseline",
            template=current_template,
            model=self.prompt_version.model,
            temperature=self.prompt_version.temperature,
            system_prompt=self.prompt_version.system_prompt,
        )
        runner = PromptDiffRunner(
            v1_prompt=v1_p,
            v2_prompt=v1_p,
            provider_v1=self.provider,
            provider_v2=self.provider,
            evaluators=self.evaluators,
            concurrency=6,
        )

        initial_report = await runner.run(self.test_cases)
        initial_passed = initial_report.aggregate_stats.get("passed_cases", 0)
        initial_pass_rate = (initial_passed / len(self.test_cases)) if self.test_cases else 1.0

        best_pass_rate = initial_pass_rate
        best_report = initial_report
        history: list[dict[str, Any]] = [
            {
                "iteration": 0,
                "template": current_template,
                "passed_cases": initial_passed,
                "pass_rate": initial_pass_rate,
            }
        ]

        if progress_cb:
            progress_cb(0, self.max_iterations, f"Initial pass rate: {initial_pass_rate * 100:.1f}%")

        # If already 100% passing, return immediately
        if initial_pass_rate >= 1.0:
            return OptimizationResult(
                original_prompt=current_template,
                optimized_prompt=current_template,
                initial_pass_rate=initial_pass_rate,
                final_pass_rate=initial_pass_rate,
                iterations=0,
                failed_cases_addressed=0,
                initial_report=initial_report,
                final_report=initial_report,
                history=history,
            )

        # Optimization loop
        for it in range(1, self.max_iterations + 1):
            failures = self._extract_failures(best_report)
            if not failures:
                break

            if progress_cb:
                progress_cb(
                    it, self.max_iterations, f"Iteration {it}: Analyzing {len(failures)} failure(s) with Meta-LLM..."
                )

            failure_text = self._format_failures_for_meta_prompt(failures)
            meta_prompt = META_OPTIMIZER_PROMPT.format(
                current_prompt=current_template,
                task_description=self.prompt_version.name or "User prompt task",
                failure_summary=failure_text,
            )

            # Generate refined candidate prompt
            if self.force_mock:
                # Deterministic optimization mock
                refined_template = f"{current_template}\n\n[INSTRUCTIONS]: Respond strictly and concisely in structured format. Answer all edge cases."
            else:
                try:
                    meta_resp = await self.meta_provider.generate(
                        prompt=meta_prompt,
                        temperature=0.3,
                        max_tokens=2048,
                    )
                    refined_template = self._parse_meta_output(meta_resp.output, current_template)
                except Exception as e:
                    logger.warning(f"Meta-LLM generation error on iteration {it}: {e}")
                    break

            # Evaluate refined candidate against baseline
            cand_p = PromptVersion(
                name=f"cand_iter_{it}",
                template=refined_template,
                model=self.prompt_version.model,
                temperature=self.prompt_version.temperature,
                system_prompt=self.prompt_version.system_prompt,
            )
            cand_runner = PromptDiffRunner(
                v1_prompt=v1_p,
                v2_prompt=cand_p,
                provider_v1=self.provider,
                provider_v2=self.provider,
                evaluators=self.evaluators,
                concurrency=6,
            )

            eval_report = await cand_runner.run(self.test_cases)
            cand_passed = eval_report.aggregate_stats.get("passed_cases", 0)
            cand_pass_rate = (cand_passed / len(self.test_cases)) if self.test_cases else 1.0

            history.append(
                {
                    "iteration": it,
                    "template": refined_template,
                    "passed_cases": cand_passed,
                    "pass_rate": cand_pass_rate,
                }
            )

            # Accept if candidate strictly improved or matched with less length
            if cand_pass_rate >= best_pass_rate:
                best_pass_rate = cand_pass_rate
                best_template = refined_template
                best_report = eval_report
                current_template = refined_template

            if best_pass_rate >= 1.0:
                break

        addressed = max(0, best_report.aggregate_stats.get("passed_cases", 0) - initial_passed)

        return OptimizationResult(
            original_prompt=self.prompt_version.template,
            optimized_prompt=best_template,
            initial_pass_rate=round(initial_pass_rate, 4),
            final_pass_rate=round(best_pass_rate, 4),
            iterations=len(history) - 1,
            failed_cases_addressed=addressed,
            initial_report=initial_report,
            final_report=best_report,
            history=history,
        )

    def save_optimized_prompt(self, template: str, output_path: str = "system_v3_optimized.txt") -> str:
        """Save optimized prompt template to file."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(template, encoding="utf-8")
        return str(target.resolve())
