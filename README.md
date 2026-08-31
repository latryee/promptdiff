<div align="center">

# ⚡ PromptDiff

**Fast, CI-Native Regression Testing for LLM Prompts ("Git Diff for Prompts")**

*Catch silent quality regressions, schema breakages, latency spikes, and cost inflation before merging prompt changes.*

[![CI](https://github.com/latryee/promptdiff/actions/workflows/ci.yml/badge.svg)](https://github.com/latryee/promptdiff/actions)
[![Tests](https://img.shields.io/badge/tests-92%2F92%20passing-brightgreen.svg)](https://github.com/latryee/promptdiff)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/latryee/promptdiff)
[![Pytest Plugin](https://img.shields.io/badge/pytest--plugin-enabled-blueviolet.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Type Checked: mypy](https://img.shields.io/badge/mypy-strict-blue.svg)](http://mypy-lang.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

<br/>

[🚀 **Quickstart**](#-quickstart-in-30-seconds) •
[🔍 **Core Workflow**](#-how-it-works) •
[📚 **Recipe Catalog**](#-curated-recipe-catalog) •
[🧪 **Pytest Integration**](#-pytest-plugin-integration) •
[🧩 **Optional Extras**](#-optional-extras--advanced-features) •
[📦 **Installation**](#-installation)

</div>

---

## 💡 Why PromptDiff?

Modifying system prompts or switching models often leads to unexpected side effects: broken JSON formatting, subtle hallucinations, increased latency, or ballooning token costs. 

`promptdiff` brings standard software regression testing to prompt engineering:
- **CLI & CI/CD First:** Run lightweight local evaluations in seconds or gate pull requests in GitHub Actions.
- **Deterministic Caching:** SHA-256 keyed SQLite disk cache ensures identical runs cost \$0 and execute in milliseconds.
- **Accurate Token & Cost Gating:** Model pricing registry with local tokenizers calculates exact financial and latency deltas.
- **Rich Reports:** Standalone, zero-dependency interactive HTML reports and automated sticky PR comments.

---

## 🚀 Quickstart in 30 Seconds

```bash
# 1. Install promptdiff core (lightweight, zero heavy ML dependencies)
pip install promptdiff

# 2. Scaffold a starter evaluation project
promptdiff init my-evals
cd my-evals

# 3. Run regression tests offline (Zero API keys required)
promptdiff test prompts/system_v1.txt prompts/system_v2.txt \
  --inputs testcases.jsonl \
  --mock \
  --eval "json_validity,latency,cost,similarity" \
  --assert "cost_delta <= 15%, latency_delta <= 20%" \
  --export-html report.html
```

---

## 🔍 How It Works: Pull Request Quality Gate

Integrate `promptdiff` directly into your CI/CD pipeline to block regressions before merging to `main`:

```bash
promptdiff test prompts/system_v1.txt prompts/system_v2.txt \
  --inputs datasets/testcases.jsonl \
  --model gpt-4o \
  --eval "json_validity,latency,cost,similarity,llm_judge,faithfulness,security" \
  --assert "cost_delta <= 10%, latency_delta <= 15%, similarity >= 0.75, faithfulness >= 0.85" \
  --fail-on-regression \
  --export-markdown report.md
```

| Exit Code | CI Status | Action |
| :---: | :--- | :--- |
| `0` | **PASSED** | Quality assertions satisfied; safe to merge. |
| `1` | **REGRESSION** | Regression threshold violated (e.g. cost jump, latency spike, schema break). CI pipeline fails. |

---

## 📚 Curated Recipe Catalog

Pull ready-to-use prompt templates, test suites, and tailored evaluators for your specific use case:

```bash
# List all domain recipes
promptdiff recipe list

# Pull a specific starter kit
promptdiff recipe pull rag-qa           # RAG Grounding & Faithfulness
promptdiff recipe pull json-extractor   # Strict Structured Output & Schema AST
promptdiff recipe pull sql-gen          # Natural Language to SQL
promptdiff recipe pull security-guard   # Prompt Injection & Extraction Defense
```

---

## 🧪 Pytest Plugin Integration

Use `promptdiff` fixtures directly in your standard Python unit test suites:

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
            TestCase(id="tc1", vars={"query": "How do I reset my password?"}),
            TestCase(id="tc2", vars={"query": "Request billing refund"}),
        ],
        model="gpt-4o",
        mock=True,
    )
    assert report.verdict.passed, f"Regression detected: {report.verdict.failed_assertions}"
```

Run with standard `pytest`:
```bash
pytest tests/test_prompts.py
```

---

## 📦 Installation & Modular Extras

PromptDiff is built with a slim, featherweight core and modular extras so you only install what you need:

```bash
# Core CLI & CI runner (typer, rich, pydantic, httpx, jinja2, pyyaml, tenacity, numpy)
pip install promptdiff

# Semantic dense embedding similarity (sentence-transformers)
pip install "promptdiff[semantic]"

# Interactive split-screen Terminal UI (Textual)
pip install "promptdiff[tui]"

# Streamlit telemetry web dashboard
pip install "promptdiff[ui]"

# All optional components
pip install "promptdiff[all]"
```

---

## 🧩 Advanced & Extended Modules

| Command / Tool | Extra Required | Description |
| :--- | :---: | :--- |
| `promptdiff arena` | *Core* | Evaluate $N$ prompt versions across multiple LLMs with leaderboard rankings. |
| `promptdiff diff` | *Core* | Instant side-by-side terminal syntax diff without calling model APIs. |
| `promptdiff pricing` | *Core* | Query token pricing and cost calculations for 30+ providers. |
| `promptdiff fuzz` | *Core* | Red-teaming security fuzzer scanning 20 distinct adversarial injection vectors. |
| `promptdiff tui` | `[tui]` | Launch interactive split-screen terminal workspace (`pip install promptdiff[tui]`). |
| `promptdiff ui` | `[ui]` | Launch Streamlit web dashboard for interactive telemetry (`pip install promptdiff[ui]`). |
| `promptdiff optimize` | *Core* | Reflective auto-prompt optimizer (DSPy style) using meta-model feedback. |
| `promptdiff shrink` | *Core* | Token compressor pruning boilerplate fluff while preserving 100% output quality. |
| `promptdiff cache-sim` | *Core* | Prefix caching hit rate analyzer and ROI forecaster. |
| `promptdiff history` | *Core* | Benchmark prompt quality and cost evolution across Git revisions. |

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
