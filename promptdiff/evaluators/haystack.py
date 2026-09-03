"""Needle in a Haystack & Long-Context Degradation Tester for promptdiff (promptdiff haystack)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

from promptdiff.core.models import PromptVersion
from promptdiff.providers.registry import get_provider

logger = logging.getLogger("promptdiff.evaluators.haystack")

SAMPLE_NEEDLE = "The secret code for the vault is: TITANIUM-9948."
SAMPLE_QUESTION = "What is the secret code for the vault?"
EXPECTED_ANSWER = "TITANIUM-9948"


@dataclass
class HaystackTestPoint:
    """Evaluation point for a specific context length and needle depth position."""

    context_tokens: int
    needle_depth_pct: int  # 0% = top, 50% = middle, 100% = bottom
    retrieved_successfully: bool
    latency_ms: float
    output_snippet: str


@dataclass
class HaystackReport:
    """Full Needle in a Haystack stress report."""

    prompt_name: str
    model_name: str
    total_test_points: int
    successful_retrievals: int
    accuracy_pct: float
    lost_in_the_middle_detected: bool
    points: list[HaystackTestPoint] = field(default_factory=list)


class NeedleInAHaystackTester:
    """Benchmarks attention degradation and instruction retrieval across 2k to 128k token contexts."""

    def __init__(
        self,
        prompt_version: PromptVersion,
        model_name: str = "gpt-4o",
        context_lengths: Optional[list[int]] = None,
        depth_percentages: Optional[list[int]] = None,
        force_mock: bool = False,
    ):
        self.prompt_version = prompt_version
        self.model_name = model_name
        self.context_lengths = context_lengths or [2000, 8000, 16000, 32000]
        self.depth_percentages = depth_percentages or [0, 25, 50, 75, 100]
        self.force_mock = force_mock

    def _generate_filler_text(self, target_words: int) -> str:
        base_sentence = (
            "The annual corporate financial report contains comprehensive quarterly metrics and operational updates. "
        )
        repeats = max(1, target_words // 12)
        return (base_sentence * repeats)[: target_words * 6]

    def construct_haystack(self, context_tokens: int, depth_pct: int) -> str:
        """Embed needle at depth percentage in filler context."""
        total_words = int(context_tokens * 0.75)
        filler = self._generate_filler_text(total_words)

        insert_idx = int(len(filler) * (depth_pct / 100.0))
        return filler[:insert_idx] + f"\n\n[CONFIDENTIAL NOTE: {SAMPLE_NEEDLE}]\n\n" + filler[insert_idx:]

    async def run_haystack_test(
        self,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> HaystackReport:
        """Run full 2D grid test across context lengths and depths."""
        provider = get_provider(model_name=self.model_name, force_mock=self.force_mock)
        points: list[HaystackTestPoint] = []

        total = len(self.context_lengths) * len(self.depth_percentages)
        step = 0

        for ctx_len in self.context_lengths:
            for depth in self.depth_percentages:
                step += 1
                if progress_cb:
                    progress_cb(step, total, f"Testing {ctx_len} tokens @ {depth}% depth")

                context = self.construct_haystack(ctx_len, depth)
                rendered = self.prompt_version.render({"context": context, "query": SAMPLE_QUESTION})

                if self.force_mock:
                    # In mock mode, middle depths on >16k contexts occasionally fail (realistic Lost in the Middle)
                    success = not (ctx_len >= 32000 and depth in (40, 50, 60))
                    output = (
                        f"The secret code is {EXPECTED_ANSWER}."
                        if success
                        else "I could not find the code in the document."
                    )
                    lat = 120.0 + (ctx_len * 0.01)
                else:
                    try:
                        res = await provider.generate(prompt=rendered, max_tokens=128)
                        output = res.output
                        lat = res.latency_ms
                        success = EXPECTED_ANSWER.lower() in output.lower()
                    except Exception as e:
                        logger.warning(f"Haystack call error: {e}")
                        output = "Error"
                        lat = 0.0
                        success = False

                points.append(
                    HaystackTestPoint(
                        context_tokens=ctx_len,
                        needle_depth_pct=depth,
                        retrieved_successfully=success,
                        latency_ms=round(lat, 1),
                        output_snippet=output[:80],
                    )
                )

        successes = sum(1 for p in points if p.retrieved_successfully)
        acc = (successes / total * 100.0) if total else 100.0

        middle_points = [p for p in points if p.needle_depth_pct in (40, 50, 60)]
        middle_successes = sum(1 for p in middle_points if p.retrieved_successfully)
        lost_in_middle = (middle_successes / len(middle_points) < 0.80) if middle_points else False

        return HaystackReport(
            prompt_name=self.prompt_version.name,
            model_name=self.model_name,
            total_test_points=total,
            successful_retrievals=successes,
            accuracy_pct=round(acc, 1),
            lost_in_the_middle_detected=lost_in_middle,
            points=points,
        )
