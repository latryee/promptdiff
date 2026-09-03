# Python SDK Reference

PromptDiff provides fully typed, async-first Python APIs for custom evaluation pipelines.

## `PromptDiffRunner`

```python
from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.core.runner import PromptDiffRunner
from promptdiff.evaluators.registry import get_evaluators
from promptdiff.providers.registry import get_provider

# Define prompt versions
v1 = PromptVersion(name="v1", template="Summarize: {{text}}")
v2 = PromptVersion(name="v2", template="Summarize concisely in bullet points: {{text}}")

# Initialize runner
runner = PromptDiffRunner(
    v1_prompt=v1,
    v2_prompt=v2,
    provider_v1=get_provider("gpt-4o", force_mock=True),
    provider_v2=get_provider("gpt-4o", force_mock=True),
    evaluators=get_evaluators(["latency", "cost", "json_validity"]),
)

report = await runner.run([TestCase(id="1", vars={"text": "PromptDiff documentation"})])
print(f"Passed: {report.verdict.passed}")
```

## Pytest Fixtures

When `promptdiff` is installed, the `pytest11` entry points are automatically registered:

### Synchronous Fixture: `promptdiff_eval`
```python
def test_prompt_regression(promptdiff_eval):
    report = promptdiff_eval(
        v1="Hello {{name}}",
        v2="Hi {{name}}!",
        mock=True,
    )
    assert report.verdict.passed is True
```

### Asynchronous Fixture: `prompt_diff`
```python
import pytest

@pytest.mark.asyncio
async def test_async_prompt_regression(prompt_diff):
    report = await prompt_diff.compare(
        v1="Hello {{name}}",
        v2="Hi {{name}}!",
        mock=True,
    )
    assert report.verdict.passed is True
```
