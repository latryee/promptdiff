"""Fine-Tuning LoRA & ChatML Dataset Synthesizer for promptdiff (promptdiff distill)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from promptdiff.core.models import DiffReport

logger = logging.getLogger("promptdiff.generators.distiller")


class FineTuningDistiller:
    """Extracts top-performing prompt evaluation runs into ChatML / OpenAI JSONL fine-tuning pairs."""

    def __init__(self, report: DiffReport, min_judge_score: float = 4.0):
        self.report = report
        self.min_judge_score = min_judge_score

    def extract_training_pairs(self) -> list[dict[str, Any]]:
        """Extract high-quality conversation pairs in standard OpenAI/ChatML JSONL format."""
        dataset = []

        for comp in self.report.comparisons:
            # Check if case passed all evaluators
            all_passed = all(s.passed for s in comp.scores.values())
            judge_score = 5.0
            if "llm_judge" in comp.scores:
                judge_score = float(comp.scores["llm_judge"].v2_score)

            if all_passed and judge_score >= self.min_judge_score:
                user_content = str(comp.test_case.vars.get("query") or comp.test_case.vars.get("input") or comp.test_case.description)
                assistant_content = comp.v2_result.output

                dataset.append({
                    "messages": [
                        {"role": "system", "content": comp.v2_result.rendered_prompt.split("\nQuery:")[0]},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ],
                    "metadata": {
                        "test_case_id": comp.test_case.id,
                        "judge_score": judge_score,
                        "cost_usd": comp.v2_result.cost_usd,
                    },
                })

        return dataset

    def export_jsonl(self, output_path: str = "distilled_train.jsonl") -> tuple[str, int]:
        """Save distilled pairs to fine-tuning JSONL file."""
        pairs = self.extract_training_pairs()
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        with target.open("w", encoding="utf-8") as f:
            for item in pairs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return str(target.resolve()), len(pairs)
