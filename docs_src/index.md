# PromptDiff Documentation

[![Version](https://img.shields.io/badge/version-v3.4.1-blue.svg)](https://github.com/latryee/promptdiff)
[![PyPI](https://img.shields.io/badge/pypi-promptdiff--eval-blue.svg)](https://pypi.org/project/promptdiff-eval/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/latryee/promptdiff/blob/main/LICENSE)

Welcome to **PromptDiff**, the enterprise-grade regression testing, cost/latency gating, and continuous evaluation framework for production LLM prompts — built to serve as **"Git Diff for Prompts"**.

> 🌐 **Interactive Browser Demo**: Test the prompt diff playground, live token cost calculator, and AST mutation visualizer without installing anything at [latryee.github.io/promptdiff](https://latryee.github.io/promptdiff/).

---

## 💡 Why PromptDiff?

Modifying system prompts or switching models often leads to unexpected side effects: broken JSON formatting, subtle hallucinations, increased latency, or ballooning token costs. 

`promptdiff` brings standard software regression testing to prompt engineering:

- **CLI & CI/CD First:** Run lightweight local evaluations in seconds or gate pull requests in GitHub Actions.
- **Deterministic Caching:** SHA-256 keyed SQLite disk cache ensures identical runs cost \$0 and execute in milliseconds.
- **Accurate Token & Cost Gating:** Model pricing registry with local tokenizers calculates exact financial and latency deltas.
- **Hardened Subprocess Sandbox:** Isolated code execution runner with resource limits and exploit-tested AST/memory sandboxing.
- **Rich Reports:** Standalone, zero-dependency interactive HTML reports and automated sticky PR comments.

---

## 🚀 Quickstart in 30 Seconds

```bash
# 1. Install promptdiff core (lightweight, zero heavy ML dependencies)
pip install promptdiff-eval

# 2. Scaffold a starter evaluation project
promptdiff init my-evals
cd my-evals

# 3. Run regression tests offline (Zero API keys required)
promptdiff test prompts/system_v1.txt prompts/system_v2.txt \
  --inputs testcases.jsonl \
  --mock \
  --eval "latency,cost,similarity" \
  --assert "cost_delta <= 15%, latency_delta <= 20%" \
  --export-html report.html
```

---

## ⚖️ Honest Comparison: PromptDiff vs Alternatives

| Feature / Dimension | **PromptDiff** | **promptfoo** | **LangSmith** | **Braintrust** |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Focus** | **Local-first regression CI/CD** & prompt version diffing | LLM red-teaming, security & multi-provider CLI evals | Production tracing, debug sessions & SaaS observability | Enterprise eval platform, proxy logging & collaboration |
| **Runtime & Language** | **Pure Python 3.10+** (zero heavy dependencies) | Node.js / TypeScript | Hosted SaaS (Python / TS SDKs) | Hosted SaaS / Enterprise on-prem |
| **Data Privacy** | **100% Local / On-prem** (SQLite on local disk; zero telemetry exfiltration) | Local / Self-hosted | Cloud SaaS (prompts & traces sent to vendor servers) | Cloud SaaS / Enterprise Private Cloud |
| **CI/CD Quality Gate** | **Native `promptdiff test` & Pytest plugin** (exit code 1 on regression) | Native CLI runner & GitHub Actions | Webhook / CI SDK assertions | CI integration via CLI / SDK |
| **Cost & Latency Diffing** | **Deterministic offline token & pricing delta engine** | Basic cost approximations | Cloud dashboard cost tracking | Cloud dashboard cost analytics |
| **Sandboxed Code Execution** | **Isolated OS subprocess** (`-I -s -B`, memory & CPU limits) | Node VM sandbox | Cloud worker execution | Cloud execution sandbox |
| **Automated Prompt Optimization** | **Reflexive meta-prompting & MCTS compiler** | Optional external scripts | Playground prompt engineering | Automated AI prompt tuner |
| **Pricing Model** | **100% Free & Open Source (MIT)** | Open Source (MIT) with Enterprise tier | Proprietary SaaS (Usage-based subscription) | Commercial SaaS / Enterprise license |

---

## 📦 Installation & Modular Extras

PromptDiff is built with a slim core and modular extras:

```bash
# Core CLI & CI runner (typer, rich, pydantic, httpx, jinja2, pyyaml, tenacity, numpy)
pip install promptdiff-eval

# Semantic dense embedding similarity (sentence-transformers)
pip install "promptdiff-eval[semantic]"

# Interactive split-screen Terminal UI (Textual)
pip install "promptdiff-eval[tui]"

# Streamlit telemetry web dashboard
pip install "promptdiff-eval[ui]"

# Documentation build dependencies (mkdocs, mkdocs-material)
pip install "promptdiff-eval[docs]"

# All optional components
pip install "promptdiff-eval[all]"
```

---

## 🧭 Navigation Guide

- **[CLI Reference](api/cli.md)**: Full command-line options for `test`, `doctor`, `arena`, `shrink`, `fuzz`, `cache-sim`, and `db`.
- **[Python SDK Reference](api/sdk.md)**: Programmatic evaluation APIs and Pytest plugin fixtures.
- **[Curated Recipes](recipes.md)**: Production starter recipes for RAG, JSON extraction, SQL generation, and security defense.
- **[Architecture Deep-Dive](architecture.md)**: System design, caching mechanics, and AST mutation engines.
- **[Security & Sandbox](security.md)**: Subprocess isolation, resource limits, and data privacy disclosure.
