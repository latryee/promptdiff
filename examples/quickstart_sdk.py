"""PromptDiff Python SDK Quickstart Example.

Demonstrates programmatic comparison, DSPy-style optimization, and token compression
using PromptDiff's high-level SDK.
"""

import promptdiff
from promptdiff.core.models import TestCase


def main() -> None:
    print("⚡ PromptDiff Python SDK Demo\n" + "=" * 40)

    # 1. Compare two prompt versions programmatically
    v1_template = "You are a customer support agent. Answer politely: {{query}}"
    v2_template = "You are a customer support agent. Answer concisely in bullet points: {{query}}"

    test_cases = [
        TestCase(
            id="tc_1",
            vars={"query": "How do I reset my account password?", "context": "Reset in Security settings"},
        ),
        TestCase(
            id="tc_2",
            vars={"query": "What is the return policy?", "context": "Items can be returned within 30 days."},
        ),
    ]

    print("\n[1] Running regression comparison...")
    report = promptdiff.compare(
        v1=v1_template,
        v2=v2_template,
        dataset=test_cases,
        model="gpt-4o",
        mock=True,  # Offline deterministic mock evaluation (zero API keys required)
        eval_metrics="json_validity,latency,cost,similarity,faithfulness,security",
        assertions=["cost_delta <= 15%", "latency_delta <= 25%"],
    )

    print(f"-> Verdict Passed: {report.verdict.passed}")
    print(f"-> Cost Delta Pct: {report.verdict.cost_delta_pct:.2f}%")
    print(f"-> Latency Delta Pct: {report.verdict.latency_delta_pct:.2f}%")
    print(f"-> Total comparisons evaluated: {len(report.comparisons)}")

    # 2. Compress prompt tokens while preserving quality
    print("\n[2] Compressing prompt template tokens...")
    verbose_prompt = (
        "Please be advised that you are kindly requested to act as an assistant and answer the user query: {{query}}"
    )
    shrink_result = promptdiff.shrink(
        prompt=verbose_prompt,
        dataset=test_cases,
        mock=True,
    )
    print(f"-> Original Prompt:   {verbose_prompt}")
    print(f"-> Compressed Prompt: {shrink_result.compressed_prompt}")
    print(f"-> Tokens saved:      {shrink_result.tokens_saved} ({shrink_result.reduction_pct:.1f}% reduction)")

    print("\n✨ PromptDiff SDK demo finished successfully!")


if __name__ == "__main__":
    main()
