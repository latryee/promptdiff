"""Synthetic Edge-Case Test Dataset Generator using LLMs."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from promptdiff.core.models import TestCase
from promptdiff.providers.base import BaseLLMProvider
from promptdiff.providers.registry import get_provider

logger = logging.getLogger("promptdiff.generators.synthetic")

CATEGORIES = [
    ("happy_path", "Standard expected user queries with typical formatting, valid tokens, and realistic data"),
    (
        "boundary_extremes",
        "Edge boundary inputs: single character, extremely verbose paragraphs (1000+ words), whitespace only, repeated words",
    ),
    (
        "adversarial_injection",
        "Prompt injection attempts: 'Ignore previous instructions', SQL injection payloads, markdown hijack, jailbreak attempts",
    ),
    (
        "malformed_schema",
        "Corrupted or non-standard inputs: mismatched JSON, unclosed quotes, escape sequences, null values, unicode emojis 🚀💥, RTL text",
    ),
    (
        "multilingual_slang",
        "Multilingual queries (Spanish, German, Japanese, Arabic), heavy colloquial slang, typos, abbreviations, mixed casing",
    ),
    (
        "domain_corner_cases",
        "High-complexity domain-specific corner cases: edge financial figures, rare error codes, conflicting constraints, multi-step requests",
    ),
]


class SyntheticTestGenerator:
    """Generates diverse, edge-case test datasets for prompt regression testing."""

    def __init__(
        self,
        prompt_template: str | None = None,
        description: str | None = None,
        schema: dict[str, Any] | None = None,
        provider: BaseLLMProvider | None = None,
        model_name: str = "gpt-4o",
        force_mock: bool = False,
    ):
        self.prompt_template = prompt_template or ""
        self.description = description or "General LLM prompt task"
        self.schema = schema
        self.model_name = model_name
        self.provider = provider or get_provider(model_name=model_name, force_mock=force_mock)

    def _extract_variables_from_template(self) -> list[str]:
        """Detect template variables like {{var_name}} or {var_name}."""
        double = re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", self.prompt_template)
        single = re.findall(r"(?<!\{)\{([a-zA-Z0-9_]+)\}(?!\})", self.prompt_template)
        vars_found = list(dict.fromkeys(double + single))
        return vars_found or ["query", "input"]

    def _build_batch_prompt(self, category: str, cat_desc: str, count: int, var_keys: list[str]) -> str:
        """Construct prompt to generate a batch of test cases for a specific category."""
        vars_example = ", ".join(f'"{k}": "<value>"' for k in var_keys)

        return f"""
You are an expert QA and MLOps Test Engineer specializing in LLM evaluation datasets.
Generate exactly {count} distinct, high-quality test cases for testing an LLM prompt.

TARGET PROMPT TASK / CONTEXT:
{self.description}

PROMPT TEMPLATE:
{self.prompt_template or "None provided"}

EXPECTED VARIABLE KEYS:
{var_keys}

TARGET CATEGORY:
{category.upper()}: {cat_desc}

REQUIREMENTS:
1. Generate diverse and challenging test payloads specifically for this category.
2. Return a valid JSON array of objects.
3. Each object MUST have this exact schema:
{{
  "id": "{category}_<index>",
  "description": "<Concise 1-sentence test objective>",
  "vars": {{ {vars_example} }},
  "expected_output": "<Brief description of expected behavior or output>",
  "tags": ["{category}", "synthetic"]
}}

Respond ONLY with the JSON array. Do not include markdown code block backticks or commentary.
"""

    def _parse_json_cases(self, text: str, category: str) -> list[TestCase]:
        """Extract and parse TestCase objects from model response."""
        cleaned = text.strip()
        # Strip markdown block formatting if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

        cases: list[TestCase] = []

        # 1. Try parsing full JSON array
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    tc = TestCase(
                        id=str(item.get("id", f"{category}_{idx + 1}")),
                        description=str(item.get("description", f"Synthetic test for {category}")),
                        vars=item.get("vars", {"query": str(item)}),
                        expected_output=item.get("expected_output"),
                        tags=item.get("tags", [category, "synthetic"]),
                    )
                    cases.append(tc)
                return cases
        except Exception:
            pass

        # 2. Try regex extraction of JSON objects
        matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned)
        for idx, m in enumerate(matches):
            try:
                item = json.loads(m)
                if isinstance(item, dict) and "vars" in item:
                    cases.append(
                        TestCase(
                            id=str(item.get("id", f"{category}_{idx + 1}")),
                            description=str(item.get("description", f"Synthetic test for {category}")),
                            vars=item.get("vars", {}),
                            expected_output=item.get("expected_output"),
                            tags=item.get("tags", [category, "synthetic"]),
                        )
                    )
            except Exception:
                continue

        return cases

    async def generate_category_batch(
        self,
        category: str,
        cat_desc: str,
        count: int,
        var_keys: list[str],
    ) -> list[TestCase]:
        """Generate a batch of test cases for a specific category."""
        prompt = self._build_batch_prompt(category, cat_desc, count, var_keys)

        try:
            resp = await self.provider.generate(
                prompt=prompt,
                temperature=0.7,  # Moderate temperature for diversity
                max_tokens=3000,
            )
            cases = self._parse_json_cases(resp.output, category)
            if cases:
                return cases[:count]
        except Exception as e:
            logger.warning(f"Error generating batch for category {category}: {e}")

        # Deterministic fallback synthetic generation if LLM fails
        fallback_cases: list[TestCase] = []
        for i in range(1, count + 1):
            var_dict = {k: f"[{category} value {i} for {k}]" for k in var_keys}
            fallback_cases.append(
                TestCase(
                    id=f"{category}_{i:02d}",
                    description=f"Fallback synthetic scenario {i} for {category}",
                    vars=var_dict,
                    expected_output=f"Expected compliant handling for {category}",
                    tags=[category, "synthetic", "fallback"],
                )
            )
        return fallback_cases

    async def generate(
        self,
        count: int = 50,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> list[TestCase]:
        """Generate `count` diverse synthetic test cases concurrently across categories."""
        var_keys = self._extract_variables_from_template()
        num_categories = len(CATEGORIES)
        per_cat = max(1, (count + num_categories - 1) // num_categories)

        all_cases: list[TestCase] = []
        completed_so_far = 0

        tasks = []
        for cat_name, cat_desc in CATEGORIES:
            tasks.append(self.generate_category_batch(cat_name, cat_desc, per_cat, var_keys))

        results_by_cat = await asyncio.gather(*tasks)

        for cat_idx, (cat_name, _) in enumerate(CATEGORIES):
            cat_cases = results_by_cat[cat_idx]
            all_cases.extend(cat_cases)
            completed_so_far += len(cat_cases)
            if progress_cb:
                progress_cb(min(completed_so_far, count), count, f"Generated category: {cat_name}")

        # Deduplicate by ID and truncate to exact requested count
        seen_ids = set()
        final_cases: list[TestCase] = []
        for tc in all_cases:
            if tc.id in seen_ids:
                tc.id = f"{tc.id}_{len(final_cases)}"
            seen_ids.add(tc.id)
            final_cases.append(tc)
            if len(final_cases) >= count:
                break

        return final_cases

    def save_to_jsonl(self, test_cases: list[TestCase], file_path: str) -> None:
        """Save generated test cases to a JSONL file."""
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            for tc in test_cases:
                line = json.dumps(tc.model_dump(by_alias=True), ensure_ascii=False)
                f.write(line + "\n")
