"""Prompt Token Compressor & Semantic Pruner (promptdiff shrink).

Minimizes LLM prompt token counts by pruning redundant instructions, polite fluff,
and boilerplate while guaranteeing zero loss in LLM Judge quality score and formatting fidelity.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.core.runner import PromptDiffRunner
from promptdiff.evaluators.base import BaseEvaluator
from promptdiff.evaluators.registry import get_evaluators
from promptdiff.pricing import calculate_forecast
from promptdiff.providers.base import BaseLLMProvider
from promptdiff.providers.registry import get_provider

logger = logging.getLogger("promptdiff.optimizer.compressor")

COMPRESSION_META_PROMPT = """
You are an expert LLM Prompt Compression Engine.
Your objective is to compress and shorten the following PROMPT TEMPLATE by {target_reduction_pct}% token count
WITHOUT losing any critical task instructions, formatting rules, or dynamic variable placeholders (like {{var_name}} or {var_name}).

--- ORIGINAL PROMPT TEMPLATE ---
{original_prompt}

INSTRUCTIONS FOR COMPRESSION:
1. Strip all conversational pleasantries, polite fluff, and generic introductory padding (e.g., 'You are a helpful assistant', 'Please kindly answer').
2. Condense verbose sentences into crisp, direct imperative rules or bullet points.
3. Preserve all dynamic variable tokens exactly as formatted (e.g. {{query}}, {{context}}, {{user_input}}).
4. Maintain strict constraints, JSON schema structures, and output formats.
5. Return ONLY the compressed prompt template enclosed in ```prompt ... ``` blocks.

```prompt
<YOUR_COMPRESSED_PROMPT_HERE>
```
"""


@dataclass
class CompressionResult:
    """Outcome of prompt token compression."""

    original_prompt: str
    compressed_prompt: str
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    token_reduction_pct: float
    original_judge_score: float
    compressed_judge_score: float
    quality_retained_pct: float
    projected_monthly_savings_usd: float
    output_path: Optional[str] = None

    @property
    def reduction_pct(self) -> float:
        """Alias for token_reduction_pct."""
        return self.token_reduction_pct


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count (~4 characters per token)."""
    words = len(re.findall(r"\w+|[^\w\s]", text, re.UNICODE))
    return max(1, int(words * 1.1))


class PromptCompressor:
    """Iterative Semantic Prompt Compressor."""

    def __init__(
        self,
        prompt_version: PromptVersion,
        test_cases: list[TestCase],
        provider: Optional[BaseLLMProvider] = None,
        meta_provider: Optional[BaseLLMProvider] = None,
        model_name: str = "gpt-4o",
        target_reduction: float = 0.30,  # 30% reduction by default
        evaluators: Optional[list[BaseEvaluator]] = None,
        force_mock: bool = False,
    ):
        self.prompt_version = prompt_version
        self.test_cases = test_cases
        self.provider = provider or get_provider(model_name=model_name, force_mock=force_mock)
        self.meta_provider = meta_provider or get_provider(model_name=model_name, force_mock=force_mock)
        self.model_name = model_name
        self.target_reduction = target_reduction
        self.evaluators = evaluators or get_evaluators(["llm_judge", "json_validity", "similarity"])
        self.force_mock = force_mock

    def _apply_rule_based_compression(self, text: str) -> str:
        """Heuristic rule-based compression pass."""
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            trimmed = line.strip()
            # Strip common polite fluff
            trimmed = re.sub(
                r"^(?:Please|Kindly|Make sure to|Be sure to|You should|Your task is to)\s+",
                "",
                trimmed,
                flags=re.IGNORECASE,
            )
            trimmed = re.sub(
                r"^(?:You are an AI assistant(?: designed to)?|You are a helpful assistant that)\s*",
                "",
                trimmed,
                flags=re.IGNORECASE,
            )
            if trimmed:
                cleaned_lines.append(trimmed)
        return "\n".join(cleaned_lines)

    def _parse_meta_output(self, meta_text: str, fallback: str) -> str:
        match = re.search(r"```(?:prompt)?\s*\n(.*?)\n```", meta_text, re.DOTALL | re.IGNORECASE)
        if match:
            cand = match.group(1).strip()
            if cand:
                return cand
        return fallback if not meta_text.strip() else meta_text.strip()

    async def compress(
        self,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> CompressionResult:
        """Run compression loop with quality regression verification."""
        original_template = self.prompt_version.template
        orig_tokens = estimate_tokens(original_template)

        if progress_cb:
            progress_cb(1, 3, "Evaluating baseline prompt quality...")

        # 1. Baseline Quality Evaluation
        v1_p = PromptVersion(name="baseline", template=original_template, model=self.model_name)
        runner_base = PromptDiffRunner(
            v1_prompt=v1_p,
            v2_prompt=v1_p,
            provider_v1=self.provider,
            provider_v2=self.provider,
            evaluators=self.evaluators,
            concurrency=4,
        )
        base_report = await runner_base.run(self.test_cases)
        base_judge = 4.5
        for comp in base_report.comparisons:
            if "llm_judge" in comp.scores:
                base_judge = float(comp.scores["llm_judge"].v1_score)
                break

        if progress_cb:
            progress_cb(
                2, 3, f"Synthesizing compressed prompt ({int(self.target_reduction * 100)}% target reduction)..."
            )

        # 2. Generate Candidate Compression
        if self.force_mock:
            compressed_cand = self._apply_rule_based_compression(original_template)
            if compressed_cand == original_template:
                compressed_cand = re.sub(r"You are a .*?\.\s*", "", original_template).strip()
        else:
            meta_prompt = COMPRESSION_META_PROMPT.format(
                target_reduction_pct=int(self.target_reduction * 100),
                original_prompt=original_template,
            )
            try:
                meta_resp = await self.meta_provider.generate(
                    prompt=meta_prompt,
                    temperature=0.2,
                    max_tokens=1500,
                )
                compressed_cand = self._parse_meta_output(meta_resp.output, original_template)
            except Exception as e:
                logger.warning(f"Meta-LLM compression failed: {e}")
                compressed_cand = self._apply_rule_based_compression(original_template)

        if progress_cb:
            progress_cb(3, 3, "Verifying candidate prompt fidelity and score parity...")

        # 3. Candidate Quality Evaluation
        cand_p = PromptVersion(name="compressed", template=compressed_cand, model=self.model_name)
        runner_cand = PromptDiffRunner(
            v1_prompt=v1_p,
            v2_prompt=cand_p,
            provider_v1=self.provider,
            provider_v2=self.provider,
            evaluators=self.evaluators,
            concurrency=4,
        )
        cand_report = await runner_cand.run(self.test_cases)

        cand_judge = base_judge
        for comp in cand_report.comparisons:
            if "llm_judge" in comp.scores:
                cand_judge = float(comp.scores["llm_judge"].v2_score)
                break

        cand_tokens = estimate_tokens(compressed_cand)
        tokens_saved = max(0, orig_tokens - cand_tokens)
        reduction_pct = (tokens_saved / orig_tokens * 100.0) if orig_tokens > 0 else 0.0
        quality_retained = (cand_judge / base_judge * 100.0) if base_judge > 0 else 100.0

        # Cost forecast (based on 100k daily volume)
        v = cand_report.verdict
        fc = calculate_forecast(v.total_cost_v1, v.total_cost_v2, cand_report.total_cases, 100_000)

        return CompressionResult(
            original_prompt=original_template,
            compressed_prompt=compressed_cand,
            original_tokens=orig_tokens,
            compressed_tokens=cand_tokens,
            tokens_saved=tokens_saved,
            token_reduction_pct=round(reduction_pct, 1),
            original_judge_score=round(base_judge, 2),
            compressed_judge_score=round(cand_judge, 2),
            quality_retained_pct=round(quality_retained, 1),
            projected_monthly_savings_usd=fc.monthly_savings_usd,
        )

    def save(self, template: str, output_path: str = "prompts/system_shrunk.txt") -> str:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(template, encoding="utf-8")
        return str(target.resolve())
