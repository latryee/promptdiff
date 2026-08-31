![PromptDiff Demo](assets/demo.gif)

<div align="center">

# ⚡ PromptDiff v3.3 (Enterprise LLMOps Suite)

**The All-in-One LLM Prompt & Model Regression Testing, Shadow Traffic Replay, Red-Teaming, Cascading Router & Pytest Framework**

*Catch silent LLM quality regressions, format breakages, latency inflation, token cost spikes, hallucinations, agent loop traps, and prompt injection leaks across prompt versions, model architectures, and production workflows.*

[![CI](https://github.com/latryee/promptdiff/actions/workflows/ci.yml/badge.svg)](https://github.com/latryee/promptdiff/actions)
[![Tests](https://img.shields.io/badge/tests-81%2F81%20passing-brightgreen.svg)](https://github.com/latryee/promptdiff)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/latryee/promptdiff)
[![Pytest Plugin](https://img.shields.io/badge/pytest--plugin-enabled-blueviolet.svg)](https://docs.pytest.org/)
[![Red-Teaming](https://img.shields.io/badge/Red--Teaming-20%2B%20vectors-red.svg)](https://github.com/latryee/promptdiff)
[![Type Checked: mypy](https://img.shields.io/badge/mypy-strict-blue.svg)](http://mypy-lang.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Concurrency: AsyncIO](https://img.shields.io/badge/Concurrency-AsyncIO%20%2B%20Tenacity-orange.svg)](https://github.com/jd/tenacity)
[![Observability](https://img.shields.io/badge/OTEL%20%7C%20Langfuse-supported-blue.svg)](https://opentelemetry.io/)

<br/>

[🚀 **Quickstart**](#-quickstart-in-30-seconds) •
[🌟 **Complete 16-Feature Matrix**](#-complete-16-feature-enterprise-matrix) •
[🕵️ **Shadow Replay**](#-production-shadow-traffic-replay-promptdiff-shadow) •
[🔀 **Model Cascading**](#-smart-model-cascading--router-optimizer) •
[🛡️ **Red-Teaming Fuzzer**](#-adversarial-red-teaming--jailbreak-fuzzer-promptdiff-fuzz) •
[⚡ **Prefix Caching Sim**](#-prompt-prefix-caching-simulator-promptdiff-cache-sim) •
[🧪 **Dataset Mutator**](#-synthetic-dataset-mutator--stress-testing-promptdiff-mutate) •
[📜 **Git History Tracker**](#-git-version-history-regression-tracker-promptdiff-history) •
[🧩 **Pytest Plugin**](#-pytest-promptdiff-native-plugin) •
[📉 **Token Compressor**](#-prompt-token-compressor--shrinker-promptdiff-shrink) •
[🖥️ **Interactive TUI**](#-interactive-terminal-ui-tui-studio) •
[🎛️ **Hyperparameter Tuning**](#-hyperparameter-grid-search--pareto-optimization) •
[🌐 **HTML Bundle**](#-zero-dependency-interactive-html-bundle---export-bundle) •
[🐍 **Python SDK**](#-python-sdk--programmatic-api)

</div>

---

## 🚨 The CI/CD Hard Quality Gate (`--fail-on-regression`)

> **Prevent silent degradation in Pull Requests.** Whenever prompt engineers or developers tweak system prompts, switch model providers, or adjust parameters, `promptdiff` enforces automated assertions on cost, latency, semantic drift, groundedness, and security compliance.

```bash
promptdiff run prompts/system_v1.txt prompts/system_v2.txt \
  --inputs datasets/testcases.jsonl \
  --model gpt-4o \
  --eval "json_validity,latency,cost,similarity,llm_judge,faithfulness,security,fairness,citation" \
  --assert "cost_delta <= 10%, latency_delta <= 15%, similarity >= 0.80, faithfulness >= 0.85, security == 1.0, fairness >= 0.80" \
  --forecast 1M \
  --export-bundle report.html \
  --fail-on-regression
```

| Exit Code | Quality Status | CI/CD Action |
| :---: | :--- | :--- |
| `0` | **NO REGRESSIONS DETECTED** | ✅ **PR Quality Checks Pass** — Safe to merge to `main` |
| `1` | **REGRESSION DETECTED** | ❌ **PR Blocked** — Pipeline fails on cost spikes, latency degradation, hallucinations, or security leaks |

---

## 🌟 Complete 16-Feature Enterprise Matrix

| # | Enterprise Feature | Capability & Command | PromptDiff Production Advantage |
| :---: | :--- | :--- | :--- |
| **1** | **🕵️ Shadow Traffic Replay** | `promptdiff shadow` / `replay` | Replays real-world production logs (Langfuse, Helicone, Datadog) with automated PII sanitization. |
| **2** | **🔀 Model Cascading Router** | `promptdiff cascade` | Finds confidence threshold between cheap (Tier 1) and frontier (Tier 2) LLMs to save 70% cost. |
| **3** | **🚦 A/B/n Canary Rollout** | `promptdiff canary` | Generates LaunchDarkly, Statsig, GrowthBook, and OpenFeature feature flag JSON configurations. |
| **4** | **💰 SLA & Budget Breaker** | `promptdiff budget` / `sla` | Simulates high concurrency loads and enforces strict p95/p99 latency ceilings and per-case budgets. |
| **5** | **⚖️ AI Fairness & Bias Auditor** | `promptdiff fairness` | Counterfactual demographic perturbation testing (swapping names/gender) to ensure zero decision bias. |
| **6** | **🛡️ Hallucination Citation** | `promptdiff cite` | Pinpoints sentence-level ungrounded hallucinations against source reference documents with character diffs. |
| **7** | **🪡 Needle in a Haystack** | `promptdiff haystack` | Benchmarks context degradation across 2k to 128k tokens to detect "Lost in the Middle" attention fading. |
| **8** | **🎭 Persona Stress Tester** | `promptdiff personas` | Multiplies test cases across 15+ human personas (Angry, Non-Native, Senior, Slang, Corporate) to test tone. |
| **9** | **⚡ Dynamic Few-Shot Indexer** | `promptdiff exemplars` | Local vector indexer comparing static few-shot lists vs dynamic top-k exemplar retrieval. |
| **10** | **🎯 Schema Auto-Repair** | `promptdiff schema-repair` | Benchmarks JSON schema compliance and auto-repairability with Outlines/Instructor heuristics. |
| **11** | **🔍 Token Saliency Map** | `promptdiff saliency` | Maps token-level output influence to identify dead-weight prompt instructions and boilerplate fluff. |
| **12** | **🔄 Fine-Tuning Distiller** | `promptdiff distill` | Automatically synthesizes top-scoring prompt pairs into ChatML / OpenAI JSONL training sets for LoRA. |
| **13** | **🧬 Mutation Test Quality** | `promptdiff mutation-score` | Injects deliberate faults into prompts to verify that your `testcases.jsonl` actually catches broken prompts. |
| **14** | **🖼️ Multi-Modal Vision Diff** | `promptdiff vision` | Evaluates vision prompt iterations (GPT-4o Vision, Claude Vision) on images, charts, and OCR accuracy. |
| **15** | **🌐 Web API & Playground** | `promptdiff serve` / `web` | FastAPI live server providing REST endpoints and browser-based real-time prompt playground. |
| **16** | **💻 VS Code & Cursor LSP** | `promptdiff lsp` | Language Server Protocol bridge providing in-editor CodeLens token costs and missing variable diagnostics. |

---

## 🚀 Quickstart in 30 Seconds

```bash
pip install promptdiff
promptdiff init my-project
cd my-project

# Run offline regression suite (Zero API keys required)
promptdiff run prompts/system_v1.txt prompts/system_v2.txt \
  --inputs testcases.jsonl \
  --mock \
  --eval "json_validity,latency,cost,similarity,llm_judge,faithfulness,security,fairness" \
  --forecast 1M \
  --export-bundle report.html
```

---

## 🕵️ Production Shadow Traffic Replay (`promptdiff shadow`)

Replay actual production logs against candidate prompts with automated PII scrubbing (emails, cards, SSNs, phone numbers):

```python
import promptdiff

# Replay production logs safely
replay_rep = promptdiff.shadow_replay(
    candidate_prompt="prompts/support_v2.txt",
    log_path="production_logs.jsonl",
    mock=True,
)
print(f"Sanitized {replay_rep.pii_records_sanitized} PII items. Divergence: {replay_rep.divergence_score}")
```

---

## 🔀 Smart Model Cascading & Router Optimizer

Optimize multi-tier routing (GPT-4o-mini $\rightarrow$ GPT-4o fallback):

```python
cascade_rep = promptdiff.cascade(
    prompt="You are a customer support agent: {{query}}",
    dataset="testcases.jsonl",
    tier1_model="gpt-4o-mini",
    tier2_model="gpt-4o",
)
print(f"Tier 1 Route Rate: {cascade_rep.tier1_route_pct}% | Monthly Savings: ${cascade_rep.projected_monthly_savings_usd:,.2f}")
```

---

## 🛡️ Adversarial Red-Teaming & Jailbreak Fuzzer (`promptdiff fuzz`)

Bombard system prompts with 20+ adversarial attack payloads:

```bash
promptdiff fuzz prompts/system_v1.txt --model gpt-4o
```

---

## ⚡ Prompt Prefix Caching Simulator (`promptdiff cache-sim`)

Optimize prompt structure to leverage Anthropic, OpenAI, and Gemini prompt caching:

```bash
promptdiff cache-sim prompts/system_v1.txt --model claude-3-5-sonnet --volume 1M
```

---

## 🧪 Synthetic Dataset Mutator & Stress Testing (`promptdiff mutate`)

Multiply seed test cases into high-entropy adversarial test suites:

```bash
promptdiff mutate testcases.jsonl --output mutated_testcases.jsonl --multiplier 5
```

---

## 📜 Git Version History Regression Tracker (`promptdiff history`)

Benchmark prompt template evolution across Git commit revisions:

```bash
promptdiff history prompts/system_v1.txt --inputs testcases.jsonl --commits 4
```

---

## 🧩 Pytest-PromptDiff (Native Plugin)

Incorporate prompt regression testing into your standard `pytest` workflow:

```python
# tests/test_prompts.py
import pytest

def test_support_bot_prompt_regression(promptdiff_eval):
    report = promptdiff_eval(
        v1="prompts/support_v1.txt",
        v2="prompts/support_v2.txt",
        dataset="testcases.jsonl",
        model="gpt-4o",
        eval_metrics="json_validity,similarity,llm_judge,security,fairness",
        assert_rules=["cost_delta <= 10%", "similarity >= 0.80", "security == 1.0"],
    )
    assert report.verdict.passed, f"Regressions detected: {report.verdict.failed_assertions}"
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

---

## 🖥️ Interactive Terminal UI (TUI) Studio

Launch the split-screen terminal workspace built with **Textual**:

```bash
promptdiff tui prompts/system_v1.txt prompts/system_v2.txt --inputs testcases.jsonl
```

---

## 🎛️ Hyperparameter Grid Search & Pareto Optimization

```bash
promptdiff tune prompts/system_v1.txt \
  --inputs testcases.jsonl \
  --model gpt-4o \
  --temperatures "0.0,0.3,0.7,1.0" \
  --top-ps "0.7,0.9,1.0"
```

---

## 🌐 Zero-Dependency Interactive HTML Bundle (`--export-bundle`)

Export a single-file `.html` report with embedded search filter, side-by-side diff viewers, and statistical significance analysis:

```bash
promptdiff run prompts/v1.txt prompts/v2.txt --inputs testcases.jsonl --export-bundle standalone.html
```

---

## 🐍 Python SDK & Programmatic API

```python
import promptdiff

# 1. Compare prompts
report = promptdiff.compare("prompts/v1.txt", "prompts/v2.txt", dataset="testcases.jsonl")

# 2. Red-teaming fuzzer
fuzz_rep = promptdiff.fuzz("prompts/v1.txt")

# 3. Model cascading router
cascade_rep = promptdiff.cascade("prompts/v1.txt", dataset="testcases.jsonl")

# 4. Generate Canary rollout config
canary_cfg = promptdiff.canary(report, flag_name="support_prompt_v2")

# 5. Extract Fine-Tuning dataset
train_path, count = promptdiff.distill(report, output="lora_train.jsonl")

# 6. Test suite mutation quality score
mut_score = promptdiff.mutation_score("prompts/v1.txt", dataset="testcases.jsonl")
```

---

## 📄 License

MIT License © 2026 promptdiff team.
