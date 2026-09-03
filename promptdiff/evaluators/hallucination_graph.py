"""Token-Level Hallucination Attribution & Bipartite Grounding Graph Evaluator.

Segments candidate outputs into fine-grained semantic claim spans, maps each span
to source context chunks via bipartite semantic bipartite matching, and colorizes
unsupported/hallucinated spans in terminal and HTML reports.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any, Optional

from promptdiff.core.models import EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.base import BaseEvaluator


@dataclass
class GroundedSpan:
    """A semantic span extracted from output text mapped to source context."""

    text: str
    start_char: int
    end_char: int
    source_context_span: Optional[str] = None
    attribution_score: float = 0.0
    status: str = "HALLUCINATED"  # GROUNDED, WEAKLY_GROUNDED, HALLUCINATED

    @property
    def is_grounded(self) -> bool:
        return self.status in ("GROUNDED", "WEAKLY_GROUNDED")


@dataclass
class HallucinationGraphResult:
    """Comprehensive graph matching and token-level attribution analysis."""

    total_tokens: int
    hallucinated_tokens: int
    token_hallucination_rate_pct: float
    grounding_accuracy_pct: float
    spans: list[GroundedSpan]
    highlighted_terminal_markup: str
    bipartite_edges: list[dict[str, Any]]


def _extract_claim_spans(text: str) -> list[tuple[str, int, int]]:
    """Segment generated response into sentence and sub-clause claim spans."""
    spans = []
    # Split by sentence boundaries, list items, or clauses
    pattern = re.compile(r"([^\.\n;\?!]+[\.\n;\?!]*)")
    for match in pattern.finditer(text):
        span_str = match.group().strip()
        if span_str and len(span_str) > 3:
            spans.append((span_str, match.start(), match.end()))

    if not spans and text.strip():
        spans.append((text.strip(), 0, len(text)))
    return spans


def _compute_span_similarity(span: str, context: str) -> tuple[float, str]:
    """Compute highest semantic alignment score and best-matching context excerpt."""
    if not context.strip() or not span.strip():
        return 0.0, ""

    span_words = set(re.findall(r"\w+", span.lower()))
    if not span_words:
        return 0.0, ""

    context_sentences = re.split(r"[\.\n\?!]+", context)
    best_score = 0.0
    best_match = ""

    for sentence in context_sentences:
        sent_clean = sentence.strip()
        if not sent_clean:
            continue
        sent_words = set(re.findall(r"\w+", sent_clean.lower()))
        if not sent_words:
            continue

        # Jaccard + sequence matcher overlap
        intersection = span_words.intersection(sent_words)
        jaccard = len(intersection) / len(span_words)
        seq_ratio = difflib.SequenceMatcher(None, span.lower(), sent_clean.lower()).ratio()

        combined_score = (jaccard * 0.7) + (seq_ratio * 0.3)
        if combined_score > best_score:
            best_score = combined_score
            best_match = sent_clean

    return min(1.0, round(best_score, 3)), best_match


class TokenAttributionEvaluator(BaseEvaluator):
    """Bipartite Graph Semantic Grounding and Token Hallucination Evaluator."""

    name: str = "token_attribution"
    description: str = "Token-level hallucination attribution and bipartite grounding graph."

    def __init__(self, ground_threshold: float = 0.65, weak_threshold: float = 0.40):
        self.ground_threshold = ground_threshold
        self.weak_threshold = weak_threshold

    def analyze(self, output_text: str, context_text: str) -> HallucinationGraphResult:
        """Analyze text and compute token-level attribution and bipartite graph edges."""
        claim_spans = _extract_claim_spans(output_text)
        grounded_spans: list[GroundedSpan] = []
        bipartite_edges: list[dict[str, Any]] = []

        total_words = max(1, len(re.findall(r"\w+", output_text)))
        hallucinated_words = 0

        markup_parts = []
        last_idx = 0

        for text_span, start, end in claim_spans:
            score, matched_ctx = _compute_span_similarity(text_span, context_text)
            words_in_span = len(re.findall(r"\w+", text_span))

            if score >= self.ground_threshold:
                status = "GROUNDED"
                tag = f"[bold green]{text_span}[/bold green]"
            elif score >= self.weak_threshold:
                status = "WEAKLY_GROUNDED"
                tag = f"[bold yellow]{text_span}[/bold yellow]"
            else:
                status = "HALLUCINATED"
                hallucinated_words += words_in_span
                tag = f"[bold red on #330000]{text_span}[/bold red on #330000]"

            g_span = GroundedSpan(
                text=text_span,
                start_char=start,
                end_char=end,
                source_context_span=matched_ctx if matched_ctx else None,
                attribution_score=score,
                status=status,
            )
            grounded_spans.append(g_span)

            if matched_ctx:
                bipartite_edges.append(
                    {
                        "output_span": text_span,
                        "context_span": matched_ctx,
                        "confidence": score,
                        "status": status,
                    }
                )

            # Assemble rich text markup
            if start > last_idx:
                markup_parts.append(output_text[last_idx:start])
            markup_parts.append(tag)
            last_idx = end

        if last_idx < len(output_text):
            markup_parts.append(output_text[last_idx:])

        hallucination_rate = round((hallucinated_words / total_words) * 100.0, 1)
        grounding_accuracy = round(100.0 - hallucination_rate, 1)

        return HallucinationGraphResult(
            total_tokens=total_words,
            hallucinated_tokens=hallucinated_words,
            token_hallucination_rate_pct=hallucination_rate,
            grounding_accuracy_pct=grounding_accuracy,
            spans=grounded_spans,
            highlighted_terminal_markup="".join(markup_parts),
            bipartite_edges=bipartite_edges,
        )

    async def async_evaluate(self, r1: RunResult, r2: RunResult, test_case: TestCase) -> EvaluatorScore:
        """Evaluate candidate output grounding against provided context."""
        context = test_case.vars.get("context", "") or test_case.vars.get("document", "")
        if not context:
            # If no context provided in testcase, check baseline response or pass
            return EvaluatorScore(
                name="token_attribution",
                passed=True,
                v1_score=1.0,
                v2_score=1.0,
                delta=0.0,
                delta_pct=0.0,
                message="No context variable provided in test case for RAG grounding verification.",
            )

        res1 = self.analyze(r1.output, context)
        res2 = self.analyze(r2.output, context)

        passed = res2.token_hallucination_rate_pct <= 15.0  # Max 15% unsupported tokens allowed
        delta = res2.grounding_accuracy_pct - res1.grounding_accuracy_pct

        return EvaluatorScore(
            name="token_attribution",
            passed=passed,
            v1_score=res1.grounding_accuracy_pct / 100.0,
            v2_score=res2.grounding_accuracy_pct / 100.0,
            delta=round(delta / 100.0, 3),
            delta_pct=round(delta, 1),
            message=(
                f"Grounding Accuracy: {res2.grounding_accuracy_pct}% "
                f"(Hallucination Rate: {res2.token_hallucination_rate_pct}%, "
                f"{len(res2.spans) - sum(1 for s in res2.spans if s.status == 'HALLUCINATED')}/{len(res2.spans)} spans supported)"
            ),
        )

    def evaluate(self, r1: RunResult, r2: RunResult, test_case: TestCase) -> EvaluatorScore:
        import asyncio

        return asyncio.run(self.async_evaluate(r1, r2, test_case))
