![PromptDiff Demo](assets/demo.gif)

<div align="center">

# ⚡ PromptDiff v3.0

**Enterprise LLM Prompt Regression Testing, Textual TUI, Hyperparameter Tuning, Token Compressor, Pytest Plugin & CI/CD PR Bot**

*Catch silent LLM quality regressions, format breakages, latency inflation, token cost spikes, hallucinations, agent loop traps, and prompt injection leaks across prompt versions, model architectures, and production workflows.*

[![CI](https://github.com/latryee/promptdiff/actions/workflows/ci.yml/badge.svg)](https://github.com/latryee/promptdiff/actions)
[![Tests](https://img.shields.io/badge/tests-59%2F59%20passing-brightgreen.svg)](https://github.com/latryee/promptdiff)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/latryee/promptdiff)
[![Pytest Plugin](https://img.shields.io/badge/pytest--plugin-enabled-blueviolet.svg)](https://docs.pytest.org/)
[![Type Checked: mypy](https://img.shields.io/badge/mypy-strict-blue.svg)](http://mypy-lang.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Concurrency: AsyncIO](https://img.shields.io/badge/Concurrency-AsyncIO%20%2B%20Tenacity-orange.svg)](https://github.com/jd/tenacity)
[![Observability](https://img.shields.io/badge/OTEL%20%7C%20Langfuse-supported-blue.svg)](https://opentelemetry.io/)

<br/>

[🚀 **Quickstart**](#-quickstart-in-30-seconds) •
[🌟 **Feature Matrix**](#-enterprise-feature-matrix) •
[🧩 **Pytest Plugin**](#-pytest-promptdiff-native-plugin) •
[📉 **Token Compressor**](#-prompt-token-compressor--shrinker-promptdiff-shrink) •
[🖥️ **Interactive TUI**](#-interactive-terminal-ui-tui-studio) •
[🎛️ **Hyperparameter Tuning**](#-hyperparameter-grid-search--pareto-optimization) •
[🤖 **Agent Trajectory**](#-multi-turn-agent-trajectory-evaluator) •
[📡 **Observability**](#-opentelemetry--langfuse-observability) •
[💰 **Cost Forecasting**](#-production-cost-forecasting-engine) •
[🤖 **GitHub PR Bot**](#-github-actions-pr-commenter-bot) •
[🐍 **Python SDK**](#-python-sdk--programmatic-api)

</div>

---

## 🚨 The CI/CD Hard Quality Gate (`--fail-on-regression`)

> **Prevent silent degradation in Pull Requests.** Whenever prompt engineers or developers tweak system prompts, switch model providers, or adjust parameters, `promptdiff` enforces automated assertions on cost, latency, semantic drift, groundedness, and security compliance.

```bash
promptdiff run prompts/system_v1.txt prompts/system_v2.txt \
  --inputs datasets/testcases.jsonl \
  --model gpt-4o \
  --eval "json_validity,latency,cost,similarity,llm_judge,faithfulness,security" \
  --assert "cost_delta <= 10%, latency_delta <= 15%, similarity >= 0.80, faithfulness >= 0.85, security == 1.0" \
  --forecast 1M \
  --fail-on-regression
```

| Exit Code | Quality Status | CI/CD Action |
| :---: | :--- | :--- |
| `0` | **NO REGRESSIONS DETECTED** | ✅ **PR Quality Checks Pass** — Safe to merge to `main` |
| `1` | **REGRESSION DETECTED** | ❌ **PR Blocked** — Pipeline fails on cost spikes, latency degradation, hallucinations, or security leaks |

---

## 🌟 Enterprise Feature Matrix

| Feature Area | Capability | PromptDiff Advantage |
| :--- | :--- | :--- |
| **🧩 Pytest Plugin** | `pytest-promptdiff` Integration | Run prompt regression assertions directly from your existing Python test suites with `@pytest.mark.promptdiff` or `promptdiff_eval`. |
| **📉 Token Compressor** | Semantic Prompt Shrinker | Prune redundant tokens & polite boilerplate by 20–40% while verifying zero quality or formatting regression (`promptdiff shrink`). |
| **🖥️ Interactive TUI** | Split-Screen Terminal Studio (`textual`) | Edit prompts side-by-side, trigger live async runs, and inspect word diffs & judge scores without leaving terminal (`promptdiff tui`). |
| **🎛️ Hyperparameter Tuning** | Grid Search & Pareto Optimization | Search temperature ($T \in [0.0, 1.0]$) & Top-P spaces to find Pareto-optimal trade-offs between quality and latency/cost (`promptdiff tune`). |
| **🤖 Agent Trajectory** | Multi-Turn Tool Chain Evaluation | Analyzes function-calling accuracy, tool argument fidelity, and infinite looping traps in agent workflows (`trajectory`). |
| **📡 Observability** | OpenTelemetry & Langfuse Exporters | Native OTLP/HTTP JSON and Langfuse event export for prompt regression traces and latency spans (`--otel`, `--langfuse`). |
| **💰 Cost Forecasting** | Production Scale Projection (`--forecast`) | Projects monthly and annual spend deltas at scale (e.g. *“Saves \$1,500/mo at 1M requests/day”*). |
| **🤖 GitHub PR Bot** | Automated Sticky PR Comments | Generates formatted Markdown summaries with status badges, metric deltas, and collapsible diffs (`scripts/pr_commenter.py`). |
| **🧠 Auto-Prompt Optimizer** | Reflective DSPy-Style Optimization | Automatically feeds failed test cases and evaluator reasoning into a Meta-LLM to rewrite prompts (`promptdiff optimize`). |
| **📚 RAG Grounding** | Faithfulness & Relevance Evaluators | Detects hallucinations against reference `context` documents and measures query intent alignment (`faithfulness`, `answer_relevance`). |
| **🛡️ Security Guardrails** | PII & Jailbreak Defense | Scans for leaked emails, credit cards (**Luhn Mod-10** verified), SSNs, API secrets, and prompt injection compromises (`security`). |
| **⚡ Async Concurrency** | Bounded Concurrent Execution | Asynchronous request multiplexing with `tenacity` exponential backoff, jitter, and automatic retry on 429/50X errors. |

---

## 🚀 Quickstart in 30 Seconds

### 1. Install via pip
```bash
pip install promptdiff
```

### 2. Scaffold a Starter Workspace
```bash
promptdiff init my-project
cd my-project
```

### 3. Run Offline Regression Test (Zero API Keys Required)
```bash
promptdiff run prompts/system_v1.txt prompts/system_v2.txt \
  --inputs testcases.jsonl \
  --mock \
  --eval "json_validity,latency,cost,similarity,llm_judge,faithfulness,security" \
  --forecast 500k
```

---

## 🧩 Pytest-PromptDiff (Native Plugin)

Incorporate prompt regression testing into your standard `pytest` workflow:

```python
# tests/test_prompts.py
import pytest
from promptdiff.core.models import TestCase

def test_support_bot_prompt_regression(promptdiff_eval):
    report = promptdiff_eval(
        v1="prompts/support_v1.txt",
        v2="prompts/support_v2.txt",
        dataset="testcases.jsonl",
        model="gpt-4o",
        eval_metrics="json_validity,similarity,llm_judge,security",
        assert_rules=["cost_delta <= 10%", "similarity >= 0.80", "security == 1.0"],
    )
    assert report.verdict.passed, f"Regressions detected: {report.verdict.failed_assertions}"
```

Run directly with pytest:
```bash
pytest tests/test_prompts.py
```

---

## 📉 Prompt Token Compressor & Shrinker (`promptdiff shrink`)

Reduce token overhead without sacrificing formatting fidelity or LLM Judge quality:

```bash
promptdiff shrink prompts/system_v1.txt \
  --inputs testcases.jsonl \
  --target-reduction 0.30 \
  --output prompts/system_shrunk.txt
```

```
                 📉 Prompt Token Compression & Quality Report                  
 Metric               Original Prompt  Compressed Prompt  Impact / Savings     
 Estimated Tokens     140 tokens       92 tokens          -34.3% (48 tokens saved)
 LLM Judge Quality    4.80 / 5.0       4.80 / 5.0         100.0% Quality Retained
 Projected Monthly    -                -                  +$450.00/mo (at 100k reqs/day)
```

---

## 🖥️ Interactive Terminal UI (TUI) Studio

Launch the split-screen terminal workspace built with **Textual**:

```bash
# Launch interactive TUI studio
promptdiff tui

# Or pre-populate prompt files
promptdiff tui prompts/system_v1.txt prompts/system_v2.txt --inputs testcases.jsonl
```

- **Split-Screen Editor**: Edit Baseline (v1) and Candidate (v2) templates side-by-side.
- **Realtime Execution**: Press `R` or click `[▶ Run Evaluation]` to execute concurrent async evaluations.
- **Tabs**: Inspect live word diffs, evaluator score tables, and projected production cost impact.

---

## 🎛️ Hyperparameter Grid Search & Pareto Optimization

Find the optimal `temperature` and `top_p` settings that maximize LLM Judge score while minimizing token cost and latency:

```bash
promptdiff tune prompts/system_v1.txt \
  --inputs testcases.jsonl \
  --model gpt-4o \
  --temperatures "0.0,0.3,0.7,1.0" \
  --top-ps "0.7,0.9,1.0"
```

```
            🎛️ Hyperparameter Grid Search & Pareto Optimal Frontier            
┌───────┬────────┬───────┬────────┬────────┬────────┬────────┬────────┬───────┐
│ Rank  │ Temp   │ Top_P │ Judge  │ Latency│ Tokens │ Cost   │ Utility│ Status│
├───────┼────────┼───────┼────────┼────────┼────────┼────────┼────────┼───────┤
│ 🥇 #1 │   0.00 │  0.70 │   4.66 │ 196.0ms│     94 │ $0.0007│ 0.5000 │ Pareto│
│ 🥈 #2 │   0.00 │  1.00 │   4.60 │ 195.8ms│     94 │ $0.0007│ 0.4938 │ Pareto│
│ 🥉 #3 │   0.50 │  0.70 │   4.46 │ 195.5ms│     94 │ $0.0007│ 0.4792 │ Pareto│
└───────┴────────┴───────┴────────┴────────┴────────┴────────┴┴────────┴───────┘
```

---

## 🤖 Multi-Turn Agent Trajectory Evaluator

Test function-calling agents and verify that tool call chains adhere to expected behavior:

```bash
promptdiff run prompts/agent_v1.txt prompts/agent_v2.txt \
  --inputs agent_testcases.jsonl \
  --eval "trajectory,cost,latency"
```

---

## 📡 OpenTelemetry & Langfuse Observability

Stream evaluation telemetry directly to your observability stack:

```bash
# Export traces to local OpenTelemetry collector / Jaeger / Datadog
promptdiff run prompts/v1.txt prompts/v2.txt --inputs testcases.jsonl --otel

# Export traces and score telemetry to Langfuse
promptdiff run prompts/v1.txt prompts/v2.txt --inputs testcases.jsonl --langfuse
```

---

## 💰 Production Cost Forecasting Engine

Pass `--forecast <daily_volume>` to project monthly and annual token expenditure variances at enterprise scale:

```bash
promptdiff run prompts/system_v1.txt prompts/system_v2.txt \
  --inputs testcases.jsonl \
  --forecast 1M
```

```
┌──────────────────────── Cost Forecasting Engine ────────────────────────┐
│                                                                         │
│  💰 Production Volume Cost Impact Forecast (1,000,000 reqs/day)         │
│                                                                         │
│  • Baseline Projected Monthly Cost:  $3,000.00                          │
│  • Candidate Projected Monthly Cost: $1,500.00                          │
│  • Projected Savings: $1,500.00/mo ($18,000.00/yr) at 1M reqs/day (-50%)│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 GitHub Actions PR Commenter Bot

Integrate `scripts/pr_commenter.py` into your GitHub Actions workflow to post or update a sticky quality gate report directly on Pull Requests:

```yaml
name: Prompt Regression CI

on:
  pull_request:
    paths:
      - 'prompts/**'
      - 'datasets/**'

jobs:
  promptdiff-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: pip install promptdiff

      - name: Run Regression Suite
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          promptdiff run prompts/system_v1.txt prompts/system_v2.txt \
            --inputs testcases.jsonl \
            --model gpt-4o \
            --eval "json_validity,latency,cost,similarity,llm_judge,faithfulness,security" \
            --assert "cost_delta <= 10%, latency_delta <= 20%, faithfulness >= 0.85, security == 1.0" \
            --forecast 1M \
            --export-json report.json \
            --fail-on-regression

      - name: Post PR Quality Report
        if: always()
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          python scripts/pr_commenter.py --report report.json --forecast 1M
```

---

## 🐍 Python SDK & Programmatic API

```python
import promptdiff

# 1. Compare two prompt versions programmatically
report = promptdiff.compare(
    v1="Answer politely: {{query}}",
    v2="Answer directly using bullet points: {{query}}",
    dataset=[{"id": "tc1", "vars": {"query": "How do I upgrade plan?"}}],
    model="gpt-4o",
    eval_metrics="json_validity,similarity,llm_judge,security",
)
print("Quality Gate Passed:", report.verdict.passed)

# 2. Compress prompt tokens
shrunk = promptdiff.shrink("prompts/system_v1.txt", dataset="testcases.jsonl", target_reduction=0.30)
print(f"Compressed ({shrunk.token_reduction_pct}% saved):", shrunk.compressed_prompt)

# 3. Optimize prompt using DSPy reflection
opt = promptdiff.optimize("prompts/system_v1.txt", dataset="testcases.jsonl", iterations=3)
print("Optimized template:", opt.optimized_prompt)
```

---

## 📄 License

MIT License © 2026 promptdiff team.
