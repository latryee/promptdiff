"""Direct Preference Optimization (DPO) & RLHF Dataset Pair Synthesizer.

Converts PromptDiff regression evaluation runs into fine-tuning preference triplets:
{"prompt": "...", "chosen": "...", "rejected": "..."}
ready for training with Axolotl, Unsloth, and Hugging Face TRL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from promptdiff.core.models import DiffReport


@dataclass
class DPOPreferencePair:
    """A single training instance for Direct Preference Optimization (DPO)."""

    prompt: str
    chosen: str
    rejected: str
    margin_score: float
    test_case_id: str


@dataclass
class DPOSynthesisResult:
    """Outcome of DPO dataset synthesis."""

    pairs: list[DPOPreferencePair]
    total_pairs: int
    mean_margin: float

    def to_jsonl(self) -> str:
        """Export dataset as standard JSONL format."""
        lines = []
        for p in self.pairs:
            row = {"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected}
            lines.append(json.dumps(row))
        return "\n".join(lines)


class DPOSynthesizer:
    """Constructs DPO pairs from DiffReport evaluations."""

    def synthesize(self, report: DiffReport, min_delta_threshold: float = 0.05) -> DPOSynthesisResult:
        """Extract chosen (superior candidate) and rejected (inferior) pairs."""
        pairs: list[DPOPreferencePair] = []

        for comp in report.comparisons:
            # Aggregate scores for v1 and v2
            v1_scores = [s.v1_score for s in comp.scores.values()]
            v2_scores = [s.v2_score for s in comp.scores.values()]

            v1_avg = sum(v1_scores) / max(1, len(v1_scores))
            v2_avg = sum(v2_scores) / max(1, len(v2_scores))
            delta = v2_avg - v1_avg

            if abs(delta) >= min_delta_threshold:
                if delta > 0:
                    chosen = comp.v2_result.output
                    rejected = comp.v1_result.output
                else:
                    chosen = comp.v1_result.output
                    rejected = comp.v2_result.output

                pairs.append(
                    DPOPreferencePair(
                        prompt=comp.v1_result.rendered_prompt,
                        chosen=chosen,
                        rejected=rejected,
                        margin_score=round(abs(delta), 3),
                        test_case_id=comp.test_case.id,
                    )
                )

        mean_m = (sum(p.margin_score for p in pairs) / len(pairs)) if pairs else 0.0
        return DPOSynthesisResult(pairs=pairs, total_pairs=len(pairs), mean_margin=round(mean_m, 3))
