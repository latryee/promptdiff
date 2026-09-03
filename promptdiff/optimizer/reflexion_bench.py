"""Iterative Reflexion & Self-Correction Convergence Benchmark.

Evaluates multi-turn prompt reflection loops, measuring step-by-step delta accuracy,
stopping criteria efficiency, and detecting unproductive infinite hallucination cycles.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReflexionStep:
    """Telemetry for an individual self-correction turn."""

    step_number: int
    candidate_output: str
    critique_feedback: str
    step_quality_score: float
    converged: bool


@dataclass
class ReflexionConvergenceReport:
    """Outcome of self-correction reflection loop benchmarking."""

    initial_score: float
    final_score: float
    total_steps_executed: int
    optimal_stopping_step: int
    convergence_rate_per_step: float
    diminishing_returns_reached: bool
    steps: list[ReflexionStep]


class ReflexionConvergenceBenchmark:
    """Benchmarks self-refinement loops to determine optimal turn depth."""

    def evaluate_trajectory(self, step_scores: list[float]) -> ReflexionConvergenceReport:
        """Analyze sequence of scores across reflection turns."""
        if not step_scores:
            step_scores = [0.5]

        steps: list[ReflexionStep] = []
        for idx, score in enumerate(step_scores):
            steps.append(
                ReflexionStep(
                    step_number=idx + 1,
                    candidate_output=f"Output for step {idx + 1}",
                    critique_feedback=f"Critique for step {idx + 1}",
                    step_quality_score=score,
                    converged=idx > 0 and (score - step_scores[idx - 1]) < 0.02,
                )
            )

        init_s = step_scores[0]
        final_s = step_scores[-1]
        delta_tot = final_s - init_s
        rate = (delta_tot / max(1, len(step_scores) - 1)) if len(step_scores) > 1 else 0.0

        # Find best stopping step before diminishing returns
        best_step = 1
        for i in range(1, len(step_scores)):
            if step_scores[i] > step_scores[best_step - 1]:
                best_step = i + 1

        diminishing = len(step_scores) >= 3 and (step_scores[-1] <= step_scores[-2])

        return ReflexionConvergenceReport(
            initial_score=round(init_s, 2),
            final_score=round(final_s, 2),
            total_steps_executed=len(step_scores),
            optimal_stopping_step=best_step,
            convergence_rate_per_step=round(rate, 3),
            diminishing_returns_reached=diminishing,
            steps=steps,
        )
