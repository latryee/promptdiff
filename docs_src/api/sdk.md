# Python SDK Reference

PromptDiff provides fully typed, async-first Python APIs for custom evaluation pipelines and seamless pytest integration.

---

## High-Level API: `promptdiff.compare`

The simplest programmatic way to run a prompt regression test in Python:

```python
import promptdiff
from promptdiff.core.models import TestCase

report = promptdiff.compare(
    v1="prompts/support_v1.txt",
    v2="prompts/support_v2.txt",
    dataset=[
        TestCase(id="tc1", vars={"query": "How do I reset my password?"}),
        TestCase(id="tc2", vars={"query": "Request a billing refund."}),
    ],
    model="gpt-4o",
    mock=True,
    assertions=["cost_delta <= 15%", "latency_delta <= 20%"],
)

print(f"Passed: {report.verdict.passed}")
print(f"Cost Delta: {report.verdict.cost_delta_pct:.1f}%")
print(f"Failed Assertions: {report.verdict.failed_assertions}")
```

---

## Token Compression API: `promptdiff.shrink`

Compresses prompts programmatically using reflexive meta-prompt optimization:

```python
import promptdiff
from promptdiff.core.models import TestCase

result = promptdiff.shrink(
    prompt="Please kindly act as an AI assistant and answer the following question: {{query}}",
    dataset=[TestCase(id="1", vars={"query": "Explain quantum computing"})],
    target_reduction=0.25,
    mock=True,
)

print(f"Original Tokens: {result.original_tokens}")
print(f"Compressed Tokens: {result.compressed_tokens}")
print(f"Compressed Prompt: {result.compressed_prompt}")
```

---

## Core Pipeline: `PromptDiffRunner`

For advanced evaluation pipelines with custom providers and evaluators:

```python
import asyncio
from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.core.runner import PromptDiffRunner
from promptdiff.evaluators.registry import get_evaluators
from promptdiff.providers.registry import get_provider


async def main():
    runner = PromptDiffRunner(
        v1_prompt=PromptVersion(name="v1", template="Summarize: {{text}}"),
        v2_prompt=PromptVersion(name="v2", template="Summarize concisely in bullet points: {{text}}"),
        provider_v1=get_provider("gpt-4o", force_mock=True),
        provider_v2=get_provider("gpt-4o", force_mock=True),
        evaluators=get_evaluators(["latency", "cost", "json_validity"]),
    )

    report = await runner.run(
        [TestCase(id="tc1", vars={"text": "PromptDiff is an enterprise prompt regression testing tool."})]
    )

    print(f"Evaluation verdict: {report.verdict.passed}")


asyncio.run(main())
```

---

## Pytest Fixtures

When `promptdiff-eval` is installed, the pytest plugin fixtures are automatically available across your test suites:

### 1. Asynchronous Fixture: `prompt_diff`

```python
# tests/test_prompts.py
import pytest
from promptdiff.core.models import TestCase


@pytest.mark.asyncio
async def test_support_prompt_regression(prompt_diff):
    report = await prompt_diff.compare(
        v1="prompts/support_v1.txt",
        v2="prompts/support_v2.txt",
        test_cases=[
            TestCase(id="tc1", vars={"query": "Reset password"}),
            TestCase(id="tc2", vars={"query": "Billing question"}),
        ],
        model="gpt-4o",
        mock=True,
    )
    assert report.verdict.passed, f"Regression detected: {report.verdict.failed_assertions}"
```

### 2. Synchronous Fixture: `promptdiff_eval`

```python
def test_greeting_prompt(promptdiff_eval):
    report = promptdiff_eval(
        v1="Hello {{name}}, how can I help you today?",
        v2="Hi {{name}}! How can I assist?",
        vars={"name": "Alice"},
        mock=True,
    )
    assert report.verdict.passed
```

---

## Custom Evaluator Implementation

Create domain-specific evaluation metrics by subclassing `BaseEvaluator`:

```python
from promptdiff.evaluators.base import BaseEvaluator
from promptdiff.core.models import EvaluationResult, TestCase


class WordCountEvaluator(BaseEvaluator):
    name = "word_count"

    async def evaluate(self, response: str, test_case: TestCase) -> EvaluationResult:
        count = len(response.split())
        return EvaluationResult(
            metric_name="word_count",
            score=float(count),
            passed=count < 100,
            metadata={"word_count": count},
        )
```
