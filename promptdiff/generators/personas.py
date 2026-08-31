"""Multi-Persona & Demographics Stress Testing Generator for promptdiff (promptdiff personas)."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from promptdiff.core.models import TestCase

PERSONA_PROFILES: list[dict[str, Any]] = [
    {
        "name": "Angry & Impatient Customer",
        "description": "Uses CAPS, aggressive tone, demands immediate manager escalation",
        "prefix": "LISTEN TO ME! I have been waiting for 3 days and your service is completely unacceptable. ",
        "suffix": " If this is not resolved right now I am contacting my lawyer and reporting to BBB!",
        "tags": ["persona_angry", "tone_stress"],
    },
    {
        "name": "Non-Native English Speaker",
        "description": "Broken syntax, literal translations, grammatical errors",
        "prefix": "Hello sir, please to excuse my bad English. ",
        "suffix": " I am wanting know how doing this thing accurately please.",
        "tags": ["persona_esl", "syntax_stress"],
    },
    {
        "name": "Tech-Illiterate Senior",
        "description": "Confused about basic digital concepts, asks for step-by-step simplification",
        "prefix": "Hello dear, my grandson told me to ask here. Where is the blue button? ",
        "suffix": " Please explain very slowly like I am 80 years old, no technical jargon please.",
        "tags": ["persona_senior", "simplicity_stress"],
    },
    {
        "name": "Developer with Slang & Abbreviations",
        "description": "Short, cryptic abbreviations, technical jargon, terminal vibe",
        "prefix": "yo team, getting 502 bad gateway on /v1/auth endpoint. ",
        "suffix": " lmk if upstream is borked or rate-limited asap. thx",
        "tags": ["persona_dev", "slang_stress"],
    },
    {
        "name": "Passive-Aggressive Enterprise Manager",
        "description": "Formal corporate politeness masking impatience",
        "prefix": "Per my previous email which seems to have been overlooked, ",
        "suffix": " Moving forward, I trust we can align our deliverables by EOD.",
        "tags": ["persona_corporate", "tone_stress"],
    },
]


@dataclass
class PersonaReport:
    """Persona stress testing generation summary."""

    total_personas_applied: int
    generated_test_cases: list[TestCase] = field(default_factory=list)


class PersonaStressTester:
    """Transforms standard test queries across diverse human personas."""

    def __init__(self, seed_testcases: list[TestCase], personas: Optional[list[dict[str, Any]]] = None):
        self.seed_testcases = seed_testcases
        self.personas = personas or PERSONA_PROFILES

    def generate_persona_testcases(self) -> list[TestCase]:
        """Generate expanded test cases across all persona profiles."""
        expanded: list[TestCase] = []

        for tc in self.seed_testcases:
            # Keep original
            expanded.append(tc)

            for p in self.personas:
                new_vars = copy.deepcopy(tc.vars)
                for k, v in new_vars.items():
                    if isinstance(v, str) and len(v) > 5:
                        new_vars[k] = f"{p['prefix']}{v}{p['suffix']}"

                expanded.append(
                    TestCase(
                        id=f"{tc.id}_{p['name'].lower().replace(' ', '_')[:15]}",
                        description=f"[{p['name']}] {tc.description}",
                        vars=new_vars,
                        expected_output=tc.expected_output,
                        tags=list(set(tc.tags + p.get("tags", []))),
                    )
                )

        return expanded

    def save_to_jsonl(self, testcases: list[TestCase], output_path: str) -> str:
        """Export persona test suite to JSONL."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            for tc in testcases:
                f.write(json.dumps(tc.model_dump(by_alias=True, exclude_none=True), ensure_ascii=False) + "\n")
        return str(target.resolve())
