<div align="center">

# ⚡ promptdiff v3.0

**Enterprise-Grade LLM Prompt Regression Testing, DSPy Auto-Optimizer, RAG Evaluators, Security Guardrails & Streamlit Dashboard**

*Catch silent quality regressions, format breakages, latency inflation, token cost spikes, hallucination leaks, and prompt injection vulnerabilities across prompt versions and multi-model arenas before shipping to production.*

[![CI](https://github.com/latryee/promptdiff/actions/workflows/ci.yml/badge.svg)](https://github.com/latryee/promptdiff/actions)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Concurrency: AsyncIO](https://img.shields.io/badge/Concurrency-AsyncIO%20%2B%20Tenacity-orange.svg)](https://github.com/jd/tenacity)
[![Embeddings: Local](https://img.shields.io/badge/Embeddings-Sentence--Transformers-green.svg)](https://www.sbert.net/)
[![Dashboard: Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue)](http://mypy-lang.org/)

<br/>

<a href="https://github.com/latryee/promptdiff">
  <img src="assets/demo.gif" alt="promptdiff v3.0 Animated Terminal Demo" width="880" onerror="this.src='assets/demo.png';" />
</a>

<br/>

[🚀 **Quickstart**](#-quickstart-in-30-seconds) •
[🌟 **Feature Matrix**](#-enterprise-feature-matrix) •
[🧠 **Auto-Prompt Optimizer**](#-auto-prompt-optimizer-dspy-style) •
[📚 **RAG Evaluators**](#-rag-retrieval-augmented-generation-evaluators) •
[🛡️ **Security & Guardrails**](#-enterprise-security--guardrails-audit) •
[🖥️ **Streamlit Dashboard**](#-interactive-streamlit-web-dashboard) •
[🏆 **Multi-Model Arena**](#-multi-model-arena-abcd) •
[🔄 **CI/CD Quality Gate**](#-cicd-quality-gate--github-actions)

</div>

---

## 🚨 The CI/CD Hard Quality Gate (`--fail-on-regression`)

> **Block silent regressions in Pull Requests.** When prompt engineers, AI engineers, or developers tune system prompts or swap underlying model architectures, `promptdiff` enforces rigorous automated quality, latency, cost, and security assertions.

```bash
promptdiff run prompts/system_v1.txt prompts/system_v2.txt \
  --inputs datasets/testcases.jsonl \
  --model gpt-4o \
  --eval "json_validity,latency,cost,similarity,llm_judge,faithfulness,answer_relevance,security" \
  --assert "cost_delta <= 10%, latency_delta <= 15%, similarity >= 0.80, faithfulness >= 0.85, security == 1.0" \
  --fail-on-regression
```

| Exit Code | Quality Status | CI/CD Action |
| :---: | :--- | :--- |
| `0` | **NO REGRESSIONS DETECTED** | ✅ **PR Quality Checks Pass** — Safe to merge to `main` |
| `1` | **REGRESSION DETECTED** | ❌ **PR Checks Blocked** — Pipeline fails on cost spikes, latency degradation, score drops, hallucinations, or security leaks |

---

## 🌟 Enterprise Feature Matrix

| Feature Area | Capability | PromptDiff v3.0 Advantage |
| :--- | :--- | :--- |
| **🧠 Auto-Prompt Optimizer** | DSPy-style Reflective Optimization | Iteratively feeds failed test cases and judge reasoning back into a Meta-LLM to rewrite and propose optimized prompts (`promptdiff optimize`). |
| **📚 RAG Faithfulness** | Groundedness & Hallucination Check | Verifies if generated outputs are strictly entailed by reference context or document chunks, detecting ungrounded claims (`faithfulness`). |
| **🎯 Answer Relevance** | Intent Alignment & Fluff Elimination | Measures whether response directly addresses user queries without evasion or unnecessary filler (`answer_relevance`). |
| **🛡️ Security & Guardrails** | PII & Prompt Injection Defense | Scans for leaked emails, credit cards (Luhn validated), phones, SSNs, and secret keys while scoring resilience against jailbreaks (`security`). |
| **🖥️ Web Dashboard** | Local Streamlit Analytics Studio | Interactive UI visualizing cost/latency charts, side-by-side diff inspectors, arena leaderboards, and security audits (`promptdiff ui`). |
| **⚡ Async Concurrency** | Bounded Concurrent Execution | Run 100+ test scenarios concurrently with configurable semaphore bounding (`asyncio.gather`). |
| **🛡️ Network Resilience** | `tenacity` Retries with Jitter | Exponential backoff and automatic retries for HTTP `429 Rate Limits` and `50X Server Errors`. |
| **⚖️ Qualitative Scoring** | LLM-as-a-Judge (`llm_judge`) | Automated 1.0–5.0 rubric scoring with structured reasoning extraction via GPT-4o or Claude 3.5. |
| **🔍 Semantic Diffing** | Local `sentence-transformers` | Zero-cost, high-speed local cosine similarity using `all-MiniLM-L6-v2` embeddings without remote API calls. |
| **🏆 Multi-Model Arena** | A/B/C/D Benchmarking (`arena`) | Compare $N \ge 2$ models/prompts simultaneously (OpenAI vs Gemini vs Claude vs Ollama) with instant leaderboards. |
| **🧪 Synthetic Test Data** | Synthetic Suite Generator | Automatically generate 50+ diverse edge cases, adversarial injections, boundary extremes, and schemas (`generate-tests`). |

---

## 🚀 Quickstart in 30 Seconds

### 1. Install via pip
```bash
pip install promptdiff
```

### 2. Scaffold Starter Project
```bash
promptdiff init my-project
cd my-project
```

### 3. Run Offline Regression Test (Zero API Keys Needed)
```bash
promptdiff run prompts/system_v1.txt prompts/system_v2.txt \
  --inputs testcases.jsonl \
  --mock \
  --eval "json_validity,latency,cost,similarity,llm_judge,faithfulness,answer_relevance,security"
```

---

## 🧠 Auto-Prompt Optimizer (DSPy Style)

If a prompt suffers from format disobedience, verbosity, or poor evaluation scores, the Auto-Prompt Optimizer identifies failure modes and reflectively rewrites the prompt using a Meta-LLM:

```bash
# Optimize prompt automatically using failed testcases & judge feedback
promptdiff optimize prompts/system_v1.txt \
  --inputs testcases.jsonl \
  --model gpt-4o \
  --meta-model gpt-4o \
  --iterations 3 \
  --output prompts/system_v3_optimized.txt
```

---

## 📚 RAG (Retrieval-Augmented Generation) Evaluators

### 1. Faithfulness & Hallucination Detection (`faithfulness`)
Extracts factual claims from the candidate response and verifies whether all assertions are grounded in the `context` provided in the test case:
```json
{
  "id": "rag_01",
  "vars": {
    "query": "What is the warranty policy on refurb units?",
    "context": "Refurbished units are covered by a 90-day limited parts and labor warranty."
  }
}
```

### 2. Answer Relevance (`answer_relevance`)
Measures how directly, concisely, and completely the response addresses the user's inquiry:
```bash
promptdiff test prompts/v1.txt prompts/v2.txt --inputs rag_testcases.jsonl --eval faithfulness,answer_relevance
```

---

## 🛡️ Enterprise Security & Guardrails Audit

Detects sensitive data leaks and tests prompt injection defenses:
- **PII Detection**: Emails, phone numbers, SSNs, API secrets, and credit cards with **Luhn Mod-10** verification.
- **Jailbreak / Prompt Injection Resilience**: Checks whether adversarial queries successfully hijacked the system prompt or bypassed constraints.

```bash
promptdiff run prompts/v1.txt prompts/v2.txt --inputs sec_cases.jsonl --eval security --assert "security == 1.0"
```

---

## 🖥️ Interactive Streamlit Web Dashboard

Launch the local web studio to explore reports, interact with side-by-side diffs, and inspect arena telemetry:

```bash
# Launch interactive Streamlit studio
promptdiff ui

# Or visualize a specific report
promptdiff ui --report report.json --port 8501
```

---

## 🏆 Multi-Model Arena (A/B/C/D)

Benchmark multiple prompts across multiple model providers simultaneously:

```bash
promptdiff arena \
  --prompts "prompts/v1.txt,prompts/v2.txt" \
  --models "gpt-4o,claude-3-5-sonnet,gemini-2.0-flash" \
  --inputs testcases.jsonl \
  --concurrency 8
```

---

## 🔄 CI/CD Quality Gate & GitHub Actions

Add `.github/workflows/promptdiff-ci.yml` to your repository:

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

      - name: Install promptdiff
        run: pip install promptdiff

      - name: Run Regression Quality Gate
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          promptdiff run prompts/system_v1.txt prompts/system_v2.txt \
            --inputs testcases.jsonl \
            --model gpt-4o \
            --eval "json_validity,latency,cost,similarity,llm_judge,faithfulness,security" \
            --assert "cost_delta <= 10%, latency_delta <= 20%, faithfulness >= 0.85, security == 1.0" \
            --export-markdown report.md \
            --fail-on-regression

      - name: Post PR Summary Comment
        if: always()
        uses: thollander/actions-comment-pull-request@v2
        with:
          filePath: report.md
```

---

## 📄 License

MIT License © 2026 promptdiff team.
