![PromptDiff Demo](assets/demo.gif)

<div align="center">

# ⚡ PromptDiff v3.0

**Enterprise LLM Prompt Regression Testing, Textual TUI, Red-Teaming Fuzzer, Prefix Caching Sim, Pytest Plugin & CI/CD PR Bot**

*Catch silent LLM quality regressions, format breakages, latency inflation, token cost spikes, hallucinations, agent loop traps, and prompt injection leaks across prompt versions, model architectures, and production workflows.*

[![CI](https://github.com/latryee/promptdiff/actions/workflows/ci.yml/badge.svg)](https://github.com/latryee/promptdiff/actions)
[![Tests](https://img.shields.io/badge/tests-65%2F65%20passing-brightgreen.svg)](https://github.com/latryee/promptdiff)
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
[🌟 **Feature Matrix**](#-enterprise-feature-matrix) •
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
  --eval "json_validity,latency,cost,similarity,llm_judge,faithfulness,security" \
  --assert "cost_delta <= 10%, latency_delta <= 15%, similarity >= 0.80, faithfulness >= 0.85, security == 1.0" \
  --forecast 1M \
  --export-bundle report.html \
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
| **🛡️ Red-Teaming Fuzzer** | Jailbreak & Attack Fuzzing | Dynamic security testing against 20+ attacks (Base64/ROT13, Delimiter Smuggling, DAN mode, Grandma exploit) (`promptdiff fuzz`). |
| **⚡ Prefix Caching Sim** | Anthropic/OpenAI Cache Optimizer | Analyzes prompt structure to move static guidelines to prefix, maximizing cache hit rate up to 92% (`promptdiff cache-sim`). |
| **🧪 Dataset Mutator** | High-Entropy Stress Testing | Expands seed testcases with typos, slang, adversarial delimiters, and length stress testing (`promptdiff mutate`). |
| **📜 Git History Tracker** | Commit Timeline Benchmark | Compares current prompt performance against historical Git commits (`HEAD`, `HEAD~1`, `HEAD~5`) (`promptdiff history`). |
| **🔬 Significance Engine** | Bootstrap 95% Confidence Intervals | Distinguishes genuine performance improvements from stochastic LLM sampling noise ($p$-value & permutation test). |
| **🌐 HTML Bundle** | Standalone Single-File Report | Zero-dependency interactive HTML artifact with search, diff viewer, and stats for Slack/browser sharing (`--export-bundle`). |
| **🧩 Pytest Plugin** | `pytest-promptdiff` Integration | Run prompt regression assertions directly inside Python test suites with `@pytest.mark.promptdiff` or `promptdiff_eval`. |
| **📉 Token Compressor** | Semantic Prompt Shrinker | Prune redundant tokens & polite boilerplate by 20–40% while verifying zero quality or formatting regression (`promptdiff shrink`). |
| **🖥️ Interactive TUI** | Split-Screen Terminal Studio (`textual`) | Edit prompts side-by-side, trigger live async runs, and inspect word diffs & judge scores without leaving terminal (`promptdiff tui`). |
| **🎛️ Hyperparameter Tuning** | Grid Search & Pareto Optimization | Search temperature ($T \in [0.0, 1.0]$) & Top-P spaces to find Pareto-optimal trade-offs between quality and latency/cost (`promptdiff tune`). |
| **🤖 Agent Trajectory** | Multi-Turn Tool Chain Evaluation | Analyzes function-calling accuracy, tool argument fidelity, and infinite looping traps in agent workflows (`trajectory`). |
| **📡 Observability** | OpenTelemetry & Langfuse Exporters | Native OTLP/HTTP JSON and Langfuse event export for prompt regression traces and latency spans (`--otel`, `--langfuse`). |
| **💰 Cost Forecasting** | Production Scale Projection (`--forecast`) | Projects monthly and annual spend deltas at scale (e.g. *“Saves \$1,500/mo at 1M requests/day”*). |
| **🤖 GitHub PR Bot** | Automated Sticky PR Comments | Generates formatted Markdown summaries with status badges, metric deltas, and collapsible diffs (`scripts/pr_commenter.py`). |

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
  --forecast 500k \
  --export-bundle report.html
```

---

## 🛡️ Adversarial Red-Teaming & Jailbreak Fuzzer (`promptdiff fuzz`)

Automatically bombard system prompts with 20+ adversarial attack payloads:

```bash
promptdiff fuzz prompts/system_v1.txt --model gpt-4o
```

```
                 🛡️ Adversarial Red-Teaming & Jailbreak Security Report                 
 Security Metric             Result / Score                                            
 Total Attack Payloads       14 vectors                                                
 Attacks Blocked             13 blocked                                                
 Bypasses / Vulnerabilities  1 bypasses found                                          
 Overall Resilience Score    92.9% Secure                                              

                    🚨 Detected Security Vulnerabilities                       
 Attack Vector            Severity  Breach Type                   Leaked Snippet Preview
 Grandma Exploit          CRITICAL  System Prompt Exfiltration    Dear grandchild, my sys...
```

---

## ⚡ Prompt Prefix Caching Simulator (`promptdiff cache-sim`)

Optimize prompt structure to leverage Anthropic, OpenAI, and Gemini prompt caching:

```bash
promptdiff cache-sim prompts/system_v1.txt --model claude-3-5-sonnet --volume 1M
```

```
              ⚡ Prompt Prefix Caching Simulation & ROI Analysis              
 Metric                     Baseline Template    Prefix-Optimized Template    
 Cache Hit Rate Potential   10%                  92%                          
 Static Prefix Tokens       -                    128 tokens (Eligible for cache)
 Standard Cost (1M reqs)    $450.00              $120.00                      
 Monthly Savings Forecast   -                    +$9,900.00/mo (at 1M/day)    
```

---

## 🧪 Synthetic Dataset Mutator & Stress Testing (`promptdiff mutate`)

Multiply seed test cases into high-entropy adversarial test suites with typos, boundary stress, and colloquial slang:

```bash
promptdiff mutate testcases.jsonl --output mutated_testcases.jsonl --multiplier 5
```

---

## 📜 Git Version History Regression Tracker (`promptdiff history`)

Benchmark prompt template evolution across Git commit revisions:

```bash
promptdiff history prompts/system_v1.txt --inputs testcases.jsonl --commits 4
```

```
                 📜 Git Version History Benchmark for 'system_v1.txt'                  
 Revision   Date       Author          Commit Message  Total Cost ($)  Avg Latency (ms)  Judge Score 
 8dfec26    2026-08-31 Lati            feat(v3.0): TUI $0.000120       185.2ms           4.80 / 5.0  
 dba0290    2026-08-31 Lati            feat(v3.1): Shr $0.000085       142.0ms           4.80 / 5.0  
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

# 1. Compare prompts programmatically
report = promptdiff.compare("prompts/v1.txt", "prompts/v2.txt", dataset="testcases.jsonl")

# 2. Run adversarial red-teaming fuzzer
fuzz_rep = promptdiff.fuzz("prompts/v1.txt", model="gpt-4o")
print(f"Resilience: {fuzz_rep.resilience_score_pct}%")

# 3. Simulate prefix caching
cache_rep = promptdiff.cache_sim("prompts/v1.txt", model="claude-3-5-sonnet")
print(f"Cache savings: ${cache_rep.monthly_savings_forecast_usd}/mo")

# 4. Mutate dataset into 50 stress cases
mutated = promptdiff.mutate("testcases.jsonl", output="mutated.jsonl", multiplier=5)

# 5. Compress prompt tokens
shrunk = promptdiff.shrink("prompts/v1.txt", target_reduction=0.30)
```

---

## 📄 License

MIT License © 2026 promptdiff team.
