<div align="center">

# ⚡ promptdiff

**Production-Grade LLM Prompt & Output Regression Tester CLI**

*Catch silent regressions, format breakages, latency spikes, and token inflation before pushing prompts to production.*

[![CI](https://github.com/latryee/promptdiff/actions/workflows/ci.yml/badge.svg)](https://github.com/latryee/promptdiff/actions)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://pydantic.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue)](http://mypy-lang.org/)

</div>

---

## 🎯 The Problem

When engineering LLM prompts, even small tweaks—such as changing a system rule, adjusting formatting requirements, or rewriting examples—can cause **silent production regressions**:

- ❌ **Format Regressions**: The model stops emitting valid JSON or omits mandatory schema fields.
- 💸 **Cost Spikes**: Output verbosity inflates token counts by 40%, drastically increasing API bills.
- ⏱️ **Latency Degradation**: Unintended chain-of-thought increases time-to-first-token and overall p95 latency.
- 📉 **Output Drift**: Key domain information or brand voice is dropped.

Testing prompts manually in web playgrounds is slow, unrepeatable, and disconnected from CI/CD pipelines.

## 🚀 The Solution

`promptdiff` is a developer-first command-line tool that treats **prompt engineering like software engineering**:
- **Side-by-Side Visual Diffing**: Terminal-native 2-column view (like `git diff`) highlighting exact word, line, and JSON key modifications.
- **Multi-Dimensional Metrics**: Automated evaluation of `json_validity`, `latency_delta`, `token_cost`, `similarity`, and `regex_match`.
- **CI/CD Regression Assertions**: Enforce hard thresholds (`--assert "cost_delta <= 10%, latency_delta <= 15%, json_validity == 1.0"`) with non-zero exit codes.
- **Deterministic Disk Caching**: SQLite SHA-256 caching for $0 re-runs and instant iteration.
- **Multi-Provider & Zero-Key Mock Mode**: Supports OpenAI, Anthropic Claude, Google Gemini, Ollama, and an offline deterministic `MockProvider`.
- **Multi-Format Export**: Generates standalone interactive HTML reports, GitHub PR comment Markdown, and JSON.

---

## 📸 Architecture Overview

```
                                 promptdiff CLI
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
        Prompt Version 1                              Prompt Version 2
      (Baseline Template)                           (Candidate Template)
                │                                             │
                └──────────────┬──────────────────────────────┘
                               ▼
                    Dataset / Test Cases Loader
                     (.jsonl, .yaml, .csv, .json)
                               │
                               ▼
                 Async Batch Execution Engine
               (Semaphore Concurrency + Cache)
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
          OpenAI/Claude   Gemini/Ollama   MockProvider
                               │
                               ▼
                      Evaluation Registry
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
  JSON Schema &           Latency & Cost          Text & Semantic
 Validity Checker         Delta Tracker          Similarity Engine
                               │
                               ▼
                     CI/CD Assertion Engine
                   (Threshold Pass/Fail Rules)
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
  Terminal UI            HTML Report            Markdown Summary
 (Side-by-Side Diff)   (Interactive Dark UI)  (GitHub Actions / PR)
```

---

## ⚡ Quickstart in 30 Seconds

### 1. Installation

```bash
pip install promptdiff
```

Or install from source in editable mode:
```bash
git clone https://github.com/latryee/promptdiff.git
cd promptdiff
pip install -e ".[dev]"
```

### 2. Run Instant Zero-Key Demo

You don't need any API keys to try `promptdiff`. Run our realistic offline mock engine:

```bash
promptdiff test examples/prompts/support_bot_v1.txt examples/prompts/support_bot_v2.txt \
  --inputs examples/testcases.jsonl \
  --eval "json_validity,latency,cost,similarity" \
  --mock \
  --export-html report.html
```

---

## 🛠️ CLI Usage & Command Reference

### Basic Prompt Regression Test

```bash
promptdiff test prompts/v1.txt prompts/v2.txt \
  --model gpt-4o \
  --eval "json_validity,latency,cost,similarity"
```

### Testing with a Test Dataset (JSONL / CSV / YAML)

```bash
promptdiff test prompts/v1.txt prompts/v2.txt \
  --inputs datasets/eval_cases.jsonl \
  --model claude-3-5-sonnet-latest \
  --concurrency 8
```

### CI/CD Regression Assertions

Block pull requests if cost or latency regresses, or if JSON schema validity drops:

```bash
promptdiff test prompts/v1.txt prompts/v2.txt \
  --inputs datasets/testcases.jsonl \
  --model gpt-4o \
  --assert "cost_delta <= 10%, latency_delta <= 15%, json_validity == 1.0" \
  --export-markdown report.md
```
> *Exits with `exit code 0` if all assertions pass, or `exit code 1` with detailed failure logs if a regression is detected.*

### Model Pricing Lookup

Check token costs per 1 Million tokens across 40+ models:

```bash
promptdiff pricing
# Or filter specific models:
promptdiff pricing gemini
```

### Quick Static Prompt Diff (Offline)

Diff two prompt files without invoking models:

```bash
promptdiff diff prompts/v1.txt prompts/v2.txt
```

### Scaffold a New Project

```bash
promptdiff init my-prompt-suite
```

### Cache Management

```bash
promptdiff cache stats
promptdiff cache clear
```

---

## 📊 Evaluation Metrics

| Metric Name | Evaluator Purpose | Output Range / Details |
| :--- | :--- | :--- |
| `json_validity` | Validates JSON syntax and schema compliance | `1.0` (Valid), `0.0` (Invalid), `0.5` (Schema Mismatch) |
| `latency` | Measures execution latency delta | Milliseconds delta (`-35.4ms (-15.2%)`) |
| `cost` | Computes token dollar cost from pricing tables | USD delta (`$0.0012 -> $0.0009 (-25%)`) |
| `similarity` | Measures sequence & token overlap preservation | `0.0` to `1.0` (100% Identical) |
| `regex_match` | Enforces output regex structure & mandatory keywords | `1.0` (Matched), `0.0` (Failed) |
| `length_drift` | Tracks output token & character inflation | Delta tokens and percentage drift |

---

## 🤖 Supported Providers

| Provider | Model Identifier Examples | Environment Variable |
| :--- | :--- | :--- |
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `o1-preview`, `o3-mini` | `OPENAI_API_KEY` |
| **Anthropic** | `claude-3-5-sonnet-latest`, `claude-3-5-haiku-latest`, `claude-3-opus-latest` | `ANTHROPIC_API_KEY` |
| **Google Gemini** | `gemini-2.0-flash`, `gemini-1.5-pro`, `gemini-1.5-flash` | `GEMINI_API_KEY` |
| **Ollama (Local)** | `ollama/llama3`, `ollama/mistral`, `ollama/deepseek-r1` | `OLLAMA_HOST` (Optional) |
| **OpenRouter / DeepSeek** | `deepseek-chat`, `deepseek-reasoner` | `OPENAI_BASE_URL`, `OPENAI_API_KEY` |
| **Mock (Offline)** | `mock`, `--mock` | *None (Zero API keys required)* |

---

## 🔄 CI/CD Integration (GitHub Actions)

Add `promptdiff` to your `.github/workflows/prompt-test.yml` to automatically prevent prompt regressions on pull requests:

```yaml
name: Prompt Regression CI

on:
  pull_request:
    paths:
      - 'prompts/**'
      - 'datasets/**'

jobs:
  prompt-regression:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install promptdiff
        run: pip install promptdiff

      - name: Run promptdiff Regression Suite
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          promptdiff test prompts/system_v1.txt prompts/system_v2.txt \
            --inputs datasets/testcases.jsonl \
            --model gpt-4o \
            --assert "cost_delta <= 10%, latency_delta <= 20%, json_validity == 1.0" \
            --export-markdown comment.md \
            --export-html report.html

      - name: Comment PR Summary
        if: always()
        uses: thollander/actions-comment-pull-request@v2
        with:
          filePath: comment.md
```

---

## 🧪 Development & Testing

Run unit tests, integration tests, and check test coverage:

```bash
# Run pytest with code coverage
pytest --cov=promptdiff --cov-report=term-missing

# Run linter & type checker
ruff check .
mypy promptdiff
```

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.
