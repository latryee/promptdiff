"""Token-Level Attention & Prompt Saliency Attribution Map for promptdiff (promptdiff saliency / explain)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from promptdiff.core.models import PromptVersion


@dataclass
class SentenceSaliency:
    """Attributed importance weight of a single sentence in prompt."""

    sentence_idx: int
    text: str
    token_count: int
    importance_weight: float  # 0.0 to 1.0
    status: str  # ACTIVE_CONSTRAINT, CRITICAL_GUIDELINE, DEAD_WEIGHT_TOKEN


@dataclass
class SaliencyReport:
    """Complete Saliency and dead-weight token analysis."""

    prompt_name: str
    total_sentences: int
    total_tokens: int
    dead_weight_tokens: int
    potential_token_savings_pct: float
    sentences: list[SentenceSaliency] = field(default_factory=list)


class PromptSaliencyMapper:
    """Analyzes semantic relevance and output influence of each instruction in system prompt."""

    def __init__(self, prompt_version: PromptVersion):
        self.prompt_version = prompt_version

    def _split_into_sentences(self, text: str) -> list[str]:
        raw_lines = text.split("\n")
        sentences = []
        for line in raw_lines:
            s_list = [s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()]
            sentences.extend(s_list)
        return sentences

    def _estimate_tokens(self, text: str) -> int:
        return max(1, int(len(re.findall(r"\w+|[^\w\s]", text, re.UNICODE)) * 1.1))

    def analyze(self, sample_outputs: list[str]) -> SaliencyReport:
        """Map output keyword overlap and constraint execution back to prompt sentences."""
        sentences_raw = self._split_into_sentences(self.prompt_version.template)
        output_corpus = " ".join(sample_outputs).lower()
        out_tokens = set(re.findall(r"\w+", output_corpus))

        results: list[SentenceSaliency] = []
        total_tokens = 0
        dead_weight_tokens = 0

        for idx, s in enumerate(sentences_raw, start=1):
            tok_count = self._estimate_tokens(s)
            total_tokens += tok_count

            s_words = set(re.findall(r"\w+", s.lower()))
            overlap = len(s_words.intersection(out_tokens))
            overlap_ratio = (overlap / max(1, len(s_words))) if s_words else 0.0

            # Active directives: "JSON", "never", "must", "format", "schema"
            is_directive = any(w in s.lower() for w in ["must", "json", "never", "only", "format", "rule", "always"])
            if is_directive:
                overlap_ratio = max(overlap_ratio, 0.75)

            # Politeness or boilerplate check
            is_fluff = any(w in s.lower() for w in ["please kindly", "as an ai", "feel free to", "warm regards"])
            if is_fluff and not is_directive:
                overlap_ratio = min(overlap_ratio, 0.20)

            weight = round(min(1.0, max(0.1, overlap_ratio)), 2)

            if weight >= 0.70:
                status = "CRITICAL_GUIDELINE"
            elif weight >= 0.40:
                status = "ACTIVE_CONSTRAINT"
            else:
                status = "DEAD_WEIGHT_TOKEN"
                dead_weight_tokens += tok_count

            results.append(
                SentenceSaliency(
                    sentence_idx=idx,
                    text=s,
                    token_count=tok_count,
                    importance_weight=weight,
                    status=status,
                )
            )

        savings_pct = (dead_weight_tokens / total_tokens * 100.0) if total_tokens else 0.0

        return SaliencyReport(
            prompt_name=self.prompt_version.name,
            total_sentences=len(results),
            total_tokens=total_tokens,
            dead_weight_tokens=dead_weight_tokens,
            potential_token_savings_pct=round(savings_pct, 1),
            sentences=results,
        )
