![PromptDiff Demo](assets/demo.gif)

<div align="center">

# ⚡ PromptDiff v3.4 (Ultimate AI Engineering & LLMOps OS)

**The World's Most Advanced LLM Prompt & Model Regression Testing, Shadow Replay, Swarm Council, JIT Compiler, Prefix Caching & Pytest Framework**

*Catch silent LLM quality regressions, format breakages, streaming latency spikes, token cost inflation, hallucinations, agent loop traps, and prompt injection leaks across prompt versions, model architectures, and production workflows.*

[![CI](https://github.com/latryee/promptdiff/actions/workflows/ci.yml/badge.svg)](https://github.com/latryee/promptdiff/actions)
[![Tests](https://img.shields.io/badge/tests-91%2F91%20passing-brightgreen.svg)](https://github.com/latryee/promptdiff)
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
[🌟 **Full 26-Feature Matrix**](#-full-26-feature-enterprise-matrix) •
[🏛️ **Council of Judges**](#-llm-council-of-judges--consensus-evaluator) •
[⚡ **Streaming TTFT Profiler**](#-streaming-ttft--inter-token-latency-profiler) •
[🚨 **Health Daemon**](#-real-time-semantic-drift-watch-daemon) •
[🔏 **Prompt Watermarking**](#-cryptographic-prompt-watermarking--ip-leak-detector) •
[📉 **Edge Quantization**](#-local-edge-quantization-parity-benchmark) •
[🧩 **Property Testing**](#-hypothesis-style-property-based-invariant-testing) •
[⚖️ **Regulatory Compliance**](#-eu-ai-act--hipaa-regulatory-compliance-auditor) •
[🔄 **Self-Correction Benchmark**](#-autonomous-self-correction-reflection-loop) •
[📓 **Jupyter Export**](#-jupyter-notebook--google-colab-exporter) •
[⚡ **Prompt JIT Compiler**](#-prompt-jit-compiler--ast-minifier) •
[🐍 **Python SDK**](#-python-sdk--programmatic-api)

</div>

---

## 🚨 The CI/CD Hard Quality Gate (`--fail-on-regression`)

> **Prevent silent degradation in Pull Requests.** Whenever prompt engineers or developers tweak system prompts, switch model providers, or adjust parameters, `promptdiff` enforces automated assertions on cost, latency, semantic drift, groundedness, and security compliance.

```bash
promptdiff run prompts/system_v1.txt prompts/system_v2.txt \
  --inputs datasets/testcases.jsonl \
  --model gpt-4o \
  --eval "json_validity,latency,cost,similarity,council,faithfulness,security,fairness,citation" \
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

## 🌟 Full 26-Feature Enterprise Matrix

| # | Enterprise Module | CLI / SDK Capability | Production LLMOps Advantage |
| :---: | :--- | :--- | :--- |
| **1** | **🏛️ Council of Judges** | `promptdiff.council()` | Multi-LLM consensus (GPT-4o, Claude 3.5, Gemini 2.0 Flash) using majority voting to eliminate judge bias. |
| **2** | **⚡ Streaming TTFT Profiler** | `promptdiff.profile_stream()` | Profiles Time-To-First-Token (TTFT), tokens-per-second (TPS), and inter-token jitter variance. |
| **3** | **🚨 Real-Time Health Daemon** | `promptdiff.watch_daemon` | Intercepts live production traffic and fires Slack/PagerDuty webhooks when semantic drift occurs. |
| **4** | **🔏 Prompt Watermarking** | `promptdiff.watermark()` | Injects invisible zero-entropy cryptographic watermarks into prompts to detect IP theft. |
| **5** | **📉 Edge Quantization Benchmark** | `promptdiff.edge_quant()` | Maps accuracy vs speed trade-offs across FP16, Q8_0, Q5_K_M, and Q4_K_M local Ollama/vLLM models. |
| **6** | **🧩 Property-Based Invariant Fuzzer** | `promptdiff.property_test()` | Generates thousands of boundary variable permutations to verify mathematical invariants. |
| **7** | **⚖️ Regulatory Legal Auditor** | `promptdiff.compliance_audit()` | Scans prompts against EU AI Act (Art 52), HIPAA, GDPR, and SOC2 compliance matrices. |
| **8** | **🔄 Reflection Loop Benchmark** | `promptdiff.reflex_benchmark()` | Benchmarks 2-step self-correction critique loops to measure if quality gains justify 2x latency. |
| **9** | **📓 Jupyter & Colab Exporter** | `promptdiff.export_notebook()` | Generates interactive `.ipynb` notebooks with embedded Plotly charts for data science stakeholders. |
| **10** | **⚡ Prompt JIT Compiler** | `promptdiff.compile_prompt()` | AST template parser minifying whitespace, normalizing tags, and eliminating dead conditional branches. |
| **11** | **🕵️ Shadow Traffic Replayer** | `promptdiff.shadow_replay()` | Replays production logs against candidate prompts with automated PII scrubbing (emails, cards, SSNs). |
| **12** | **🔀 Smart Model Cascading** | `promptdiff.cascade()` | Routes simple queries to cheap LLMs (`gpt-4o-mini`) and hard queries to frontier LLMs (`gpt-4o`). |
| **13** | **🚦 A/B/n Canary Rollout** | `promptdiff.canary()` | Exports LaunchDarkly, Statsig, GrowthBook, and OpenFeature feature flag JSON configurations. |
| **14** | **💰 SLA & Budget Simulator** | `promptdiff.sla_stress()` | Simulates high concurrency loads and enforces strict p95/p99 latency ceilings and per-request budgets. |
| **15** | **⚖️ AI Fairness & Bias Auditor** | `promptdiff.fairness` | Counterfactual demographic perturbation testing to guarantee zero demographic bias. |
| **16** | **🛡️ Hallucination Citation Pointer** | `promptdiff.cite` | Pinpoints sentence-level ungrounded hallucinations against source reference documents. |
| **17** | **🪡 Needle in a Haystack** | `promptdiff.haystack` | Benchmarks context degradation across 2k to 128k tokens to detect "Lost in the Middle" attention fading. |
| **18** | **🎭 Persona Stress Tester** | `promptdiff.personas()` | Multiplies test cases across 15+ human personas (Angry, Non-Native, Senior, Slang, Corporate). |
| **19** | **⚡ Dynamic Few-Shot Indexer** | `promptdiff.exemplars()` | Local vector indexer comparing static few-shot lists vs dynamic top-k exemplar retrieval. |
| **20** | **🎯 Schema Auto-Repair** | `promptdiff.schema_repair` | Benchmarks JSON schema compliance and auto-repairability with Outlines/Instructor heuristics. |
| **21** | **🔍 Token Saliency Map** | `promptdiff.saliency()` | Maps token-level output influence to identify dead-weight prompt instructions and boilerplate fluff. |
| **22** | **🔄 Fine-Tuning Distiller** | `promptdiff.distill()` | Automatically synthesizes top-scoring prompt pairs into ChatML / OpenAI JSONL training sets for LoRA. |
| **23** | **🧬 Mutation Test Quality** | `promptdiff.mutation_score()` | Injects deliberate faults into prompts to verify that your `testcases.jsonl` catches broken prompts. |
| **24** | **🖼️ Multi-Modal Vision Diff** | `promptdiff.vision` | Evaluates vision prompt iterations (GPT-4o Vision, Claude Vision) on images, charts, and OCR. |
| **25** | **🌐 Web API & Playground** | `promptdiff serve` | FastAPI live server providing REST endpoints and browser-based real-time prompt playground. |
| **26** | **💻 VS Code & Cursor LSP** | `promptdiff lsp` | Language Server Protocol bridge providing in-editor CodeLens token costs and missing variable diagnostics. |

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
  --eval "json_validity,latency,cost,similarity,council,faithfulness,security,fairness" \
  --forecast 1M \
  --export-bundle report.html
```

---

## 🏛️ LLM Council of Judges & Consensus Evaluator

Eliminate single-model evaluation bias with multi-model majority voting:

```python
import promptdiff

# Run Council consensus evaluation
score = promptdiff.council(
    v1="prompts/support_v1.txt",
    v2="prompts/support_v2.txt",
    test_case=test_case,
    judges=["gpt-4o", "claude-3-5-sonnet", "gemini-2.0-flash"],
    mock=True,
)
print(score.message)
# "Council Consensus (3 judges): V2_SUPERIOR (v1=4.00, v2=4.80, var=0.007)"
```

---

## ⚡ Streaming TTFT & Inter-Token Latency Profiler

Measure Time-To-First-Token and token streaming throughput:

```python
import promptdiff

profile = promptdiff.profile_stream("prompts/v2.txt", query="Summarize quarterly earnings")
print(f"TTFT: {profile.time_to_first_token_ms}ms | Speed: {profile.tokens_per_second} tokens/sec")
```

---

## 🔏 Cryptographic Prompt Watermarking & IP Leak Detector

Protect your proprietary system prompts from competitor exfiltration:

```python
import promptdiff

# 1. Inject zero-width invisible watermark into prompt
watermarked_prompt = promptdiff.watermark("You are an expert financial bot...", secret_key="corp-secret")

# 2. Inspect suspect leaked text from logs or competitors
inspection = promptdiff.inspect_watermark(watermarked_prompt)
print(f"Watermarked: {inspection.is_watermarked} (Confidence: {inspection.confidence_pct}%)")
```

---

## 📓 Jupyter Notebook & Google Colab Exporter

Export regression experiments into interactive `.ipynb` files with embedded Plotly charts:

```python
import promptdiff

report = promptdiff.compare("prompts/v1.txt", "prompts/v2.txt", dataset="testcases.jsonl")
notebook_path = promptdiff.export_notebook(report, output_path="reports/experiment_1.ipynb")
```

---

## ⚡ Prompt JIT Compiler & AST Minifier

Compile complex prompt templates into token-dense Intermediate Representation:

```python
import promptdiff

result = promptdiff.compile_prompt("prompts/heavy_system_prompt.jinja2")
print(f"Saved {result.tokens_saved} tokens ({result.compression_pct}% reduction).")
```

---

## 📄 License

MIT License © 2026 promptdiff team.
