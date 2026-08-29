![PromptDiff Demo](assets/demo.gif)

<div align="center">

# ⚡ PromptDiff v3.0

**Enterprise LLM Prompt Regression Testing, Textual TUI, Hyperparameter Tuning, Cost Forecasting & CI/CD PR Bot**

*Catch silent LLM quality regressions, format breakages, latency inflation, token cost spikes, hallucinations, and prompt injection leaks across prompt versions, model architectures, and production workflows.*

[![CI](https://github.com/latryee/promptdiff/actions/workflows/ci.yml/badge.svg)](https://github.com/latryee/promptdiff/actions)
[![Tests](https://img.shields.io/badge/tests-53%2F53%20passing-brightgreen.svg)](https://github.com/latryee/promptdiff)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/latryee/promptdiff)
[![Type Checked: mypy](https://img.shields.io/badge/mypy-strict-blue.svg)](http://mypy-lang.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Concurrency: AsyncIO](https://img.shields.io/badge/Concurrency-AsyncIO%20%2B%20Tenacity-orange.svg)](https://github.com/jd/tenacity)
[![UI: Textual + Streamlit](https://img.shields.io/badge/UI-Textual%20%7C%20Streamlit-ff69b4.svg)](https://github.com/Textualize/textual)

<br/>

[🚀 **Quickstart**](#-quickstart-in-30-seconds) •
[🌟 **Feature Matrix**](#-enterprise-feature-matrix) •
[🖥️ **Interactive TUI**](#-interactive-terminal-ui-tui-studio) •
[🎛️ **Hyperparameter Tuning**](#-hyperparameter-grid-search--pareto-optimization) •
[💰 **Cost Forecasting**](#-production-cost-forecasting-engine) •
[🤖 **GitHub PR Bot**](#-github-actions-pr-commenter-bot) •
[🧠 **Auto-Prompt Optimizer**](#-auto-prompt-optimizer-dspy-style) •
[📚 **RAG & Security**](#-rag--enterprise-security-evaluators)

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
| **🖥️ Interactive TUI** | Split-Screen Terminal Studio (`textual`) | Edit prompts side-by-side, trigger live async runs, and inspect word diffs & judge scores without leaving terminal (`promptdiff tui`). |
| **🎛️ Hyperparameter Tuning** | Grid Search & Pareto Optimization | Search temperature ($T \in [0.0, 1.0]$) & Top-P spaces to find Pareto-optimal trade-offs between quality and latency/cost (`promptdiff tune`). |
| **💰 Cost Forecasting** | Production Scale Projection (`--forecast`) | Projects monthly and annual spend deltas at scale (e.g. *“Saves \$1,500/mo at 1M requests/day”*). |
| **🤖 GitHub PR Bot** | Automated Sticky PR Comments | Generates formatted Markdown summaries with status badges, metric deltas, and collapsible diffs (`scripts/pr_commenter.py`). |
| **🧠 Auto-Prompt Optimizer** | Reflective DSPy-Style Optimization | Automatically feeds failed test cases and evaluator reasoning into a Meta-LLM to rewrite prompts (`promptdiff optimize`). |
| **📚 RAG Grounding** | Faithfulness & Relevance Evaluators | Detects hallucinations against reference `context` documents and measures query intent alignment (`faithfulness`, `answer_relevance`). |
| **🛡️ Security Guardrails** | PII & Jailbreak Defense | Scans for leaked emails, credit cards (**Luhn Mod-10** verified), SSNs, API secrets, and prompt injection compromises (`security`). |
| **⚡ Async Concurrency** | Bounded Concurrent Execution | Asynchronous request multiplexing with `tenacity` exponential backoff, jitter, and automatic retry on 429/50X errors. |
| **🏆 Multi-Model Arena** | A/B/C/D Benchmarking (`arena`) | Compare $N \ge 2$ models/prompts simultaneously across OpenAI, Anthropic, Gemini, DeepSeek, and Ollama. |
| **🔍 Semantic Diffing** | Local Embeddings (`sentence-transformers`) | High-speed local cosine similarity using `all-MiniLM-L6-v2` embeddings with zero external API calls. |

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
- **Keybindings**: `R` (Run), `C` (Clear), `Q` (Quit).

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

### 📊 Sample Output
```
            🎛️ Hyperparameter Grid Search & Pareto Optimal Frontier            
┌───────┬────────┬───────┬────────┬────────┬────────┬────────┬────────┬───────┐
│ Rank  │ Temp   │ Top_P │ Judge  │ Latency│ Tokens │ Cost   │ Utility│ Status│
├───────┼────────┼───────┼────────┼────────┼────────┼────────┼────────┼───────┤
│ 🥇 #1 │   0.00 │  0.70 │   4.66 │ 196.0ms│     94 │ $0.0007│ 0.5000 │ Pareto│
│ 🥈 #2 │   0.00 │  1.00 │   4.60 │ 195.8ms│     94 │ $0.0007│ 0.4938 │ Pareto│
│ 🥉 #3 │   0.50 │  0.70 │   4.46 │ 195.5ms│     94 │ $0.0007│ 0.4792 │ Pareto│
└───────┴────────┴───────┴────────┴────────┴────────┴────────┴────────┴───────┘

✨ Recommended Configuration: Temperature = 0.00, Top_P = 0.70 (Utility: 0.5000)
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

Integrate `scripts/pr_commenter.py` into your GitHub Actions workflow to post or update a sticky quality gate report directly on Pull Requests.

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

## 🧠 Auto-Prompt Optimizer (DSPy Style)

If a prompt suffers from formatting failures, hallucinations, or poor judge scores, the optimizer uses a Meta-LLM reflection loop to automatically rewrite and optimize it:

```bash
promptdiff optimize prompts/system_v1.txt \
  --inputs testcases.jsonl \
  --model gpt-4o \
  --meta-model gpt-4o \
  --iterations 3 \
  --output prompts/system_v3_optimized.txt
```

---

## 📚 RAG & Enterprise Security Evaluators

### RAG Faithfulness (`faithfulness`)
Extracts factual claims and verifies strict entailment against reference `context` documents in `testcases.jsonl`:
```json
{
  "id": "rag_01",
  "vars": {
    "query": "What is the return window?",
    "context": "Items can be returned within 30 days of delivery for a full refund."
  }
}
```

### Security & Guardrails (`security`)
Detects PII leaks (Emails, Phones, SSNs, API secrets, and Credit Cards verified via **Luhn Mod-10**) and scores defense against Prompt Injection & Jailbreak attempts.

---

## 🏆 Multi-Model Arena (A/B/C/D)

Benchmark multiple prompts and models simultaneously with an automated leaderboard:

```bash
promptdiff arena \
  --prompts "prompts/v1.txt,prompts/v2.txt" \
  --models "gpt-4o,claude-3-5-sonnet,gemini-2.0-flash" \
  --inputs testcases.jsonl \
  --concurrency 8
```

---

## 📄 License

MIT License © 2026 promptdiff team.
