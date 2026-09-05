<div align="center">

# ⚡ PromptDiff

**Fast, CI-Native Regression Testing for LLM Prompts ("Git Diff for Prompts")**

*Catch silent quality regressions, schema breakages, latency spikes, and cost inflation before merging prompt changes.*

[![Version](https://img.shields.io/badge/version-v3.4.1-blue.svg)](https://github.com/latryee/promptdiff)
[![Live Demo](https://img.shields.io/badge/live%20demo-interactive%20showcase-6366f1.svg)](https://latryee.github.io/promptdiff/)
[![Python versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://github.com/latryee/promptdiff)
[![CI](https://github.com/latryee/promptdiff/actions/workflows/ci.yml/badge.svg)](https://github.com/latryee/promptdiff/actions)
[![Tests Passing](https://raw.githubusercontent.com/latryee/promptdiff/main/.github/badges/tests.svg)](https://github.com/latryee/promptdiff/actions)
[![Coverage](https://raw.githubusercontent.com/latryee/promptdiff/main/.github/badges/coverage.svg)](https://github.com/latryee/promptdiff/actions)
[![Security: Isolated Sandbox](https://img.shields.io/badge/sandbox-isolated%20subprocess-blue.svg)](promptdiff/evaluators/code_sandbox.py)
[![Pytest Plugin](https://img.shields.io/badge/pytest--plugin-enabled-blueviolet.svg)](https://docs.pytest.org/)
[![Type Checked: mypy](https://img.shields.io/badge/mypy-strict-blue.svg)](http://mypy-lang.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<br/>

<img src="assets/demo.gif" alt="PromptDiff Interactive CLI & HTML Demo" width="850" />

<br/>
<br/>

[🌐 **Live Interactive Demo**](https://latryee.github.io/promptdiff/) •
[🚀 **Quickstart**](#-quickstart-in-30-seconds) •
[🔍 **Core Workflow**](#-how-it-works-pull-request-quality-gate) •
[🩺 **Environment Doctor**](#-environment-doctor) •
[🐍 **Python SDK**](#-python-sdk) •
[📚 **Recipe Catalog**](#-curated-recipe-catalog) •
[🧪 **Pytest Integration**](#-pytest-plugin-integration) •
[📦 **Installation**](#-installation--modular-extras)

</div>

> 🌐 **Try it Live in Your Browser:** Test the interactive prompt diff playground, token cost calculator, and AST mutation visualizer without installing anything: **[latryee.github.io/promptdiff](https://latryee.github.io/promptdiff/)**

---

## 📑 Table of Contents

- [💡 Why PromptDiff?](#-why-promptdiff)
- [⚖️ Honest Comparison: PromptDiff vs Alternatives](#-honest-comparison-promptdiff-vs-alternatives)
- [🚀 Quickstart in 30 Seconds](#-quickstart-in-30-seconds)
- [🩺 Environment Doctor](#-environment-doctor)
- [🔍 How It Works: Pull Request Quality Gate](#-how-it-works-pull-request-quality-gate)
- [📚 Curated Recipe Catalog](#-curated-recipe-catalog)
- [🧪 Pytest Plugin Integration](#-pytest-plugin-integration)
- [🐍 Python SDK](#-python-sdk)
- [📦 Installation & Modular Extras](#-installation--modular-extras)
- [🧩 Advanced & Extended Modules](#-advanced--extended-modules)
- [🛡️ Production Engineering Standards & Quality Assurance](#-production-engineering-standards--quality-assurance)
- [🏛️ Architecture Deep-Dive](#-architecture-deep-dive)
- [🔒 Data Privacy & Local Storage Disclosure](#-data-privacy--local-storage-disclosure)
- [🤝 Community & Contributing](#-community--contributing)
- [📈 Star History](#-star-history)
- [📄 License](#-license)

---

## 💡 Why PromptDiff?

Modifying system prompts or switching models often leads to unexpected side effects: broken JSON formatting, subtle hallucinations, increased latency, or ballooning token costs. 

`promptdiff` brings standard software regression testing to prompt engineering:
- **CLI & CI/CD First:** Run lightweight local evaluations in seconds or gate pull requests in GitHub Actions.
- **Deterministic Caching:** SHA-256 keyed SQLite disk cache ensures identical runs cost \$0 and execute in milliseconds.
- **Accurate Token & Cost Gating:** Model pricing registry with local tokenizers calculates exact financial and latency deltas.
- **Hardened Subprocess Sandbox:** Isolated code execution runner with resource limits and exploit-tested AST/memory sandboxing.
- **Rich Reports:** Standalone, zero-dependency interactive HTML reports and automated sticky PR comments ([Explore Live Demo](https://latryee.github.io/promptdiff/)).

---

## ⚖️ Honest Comparison: PromptDiff vs Alternatives

Selecting the right evaluation tool depends heavily on your team's workflow, runtime stack, and data sovereignty requirements:

| Feature / Dimension | **PromptDiff** | **promptfoo** | **LangSmith** | **Braintrust** |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Focus** | **Local-first regression CI/CD** & prompt version diffing | LLM red-teaming, security & multi-provider CLI evals | Production tracing, debug sessions & SaaS observability | Enterprise eval platform, proxy logging & collaboration |
| **Runtime & Language** | **Pure Python 3.10+** (zero heavy dependencies) | Node.js / TypeScript | Hosted SaaS (Python / TS SDKs) | Hosted SaaS / Enterprise on-prem |
| **Data Privacy** | **100% Local / On-prem** (SQLite on local disk; zero telemetry exfiltration) | Local / Self-hosted | Cloud SaaS (prompts & traces sent to vendor servers) | Cloud SaaS / Enterprise Private Cloud |
| **CI/CD Quality Gate** | **Native `promptdiff test` & Pytest plugin** (exit code 1 on regression) | Native CLI runner & GitHub Actions | Webhook / CI SDK assertions | CI integration via CLI / SDK |
| **Cost & Latency Diffing** | **Deterministic offline token & pricing delta engine** | Basic cost approximations | Cloud dashboard cost tracking | Cloud dashboard cost analytics |
| **Sandboxed Code Execution** | **Isolated OS subprocess** (`-I -s -B`, memory & CPU limits) | Node VM sandbox | Cloud worker execution | Cloud execution sandbox |
| **Automated Prompt Optimization** | **Reflexive meta-prompting & MCTS compiler** | Optional external scripts | Playground prompt engineering | Automated AI prompt tuner |
| **Full Distributed Tracing** | ⚠️ *Telemetry logs only* (OpenTelemetry / MLflow exportable) | ⚠️ *Eval-focused only* | ✅ **Full distributed waterfall traces** | ✅ **Distributed trace logging & proxy** |
| **Pricing Model** | **100% Free & Open Source (MIT)** | Open Source (MIT) with Enterprise tier | Proprietary SaaS (Usage-based subscription) | Commercial SaaS / Enterprise license |

### When to choose which:
- **Choose PromptDiff** if you are a Python/MLOps team that treats prompts as code in Git, wants pytest-native integration, requires 100% data sovereignty without external cloud dependencies, and needs fast PR regression gates.
- **Choose promptfoo** if you have a Node.js/TypeScript stack, want a rich browser-based red-teaming workspace, or need pre-packaged adversarial jailbreak test suites.
- **Choose LangSmith** if your primary requirement is distributed production trace visualization across multi-agent LangChain graphs.
- **Choose Braintrust** if you want an enterprise-managed centralized cloud evaluation platform with web-based team playground collaboration.

---

## 🚀 Quickstart in 30 Seconds

> [!NOTE]
> **PyPI Package Name**: Published on PyPI as [`promptdiff-eval`](https://pypi.org/project/promptdiff-eval/) due to a legacy package name collision, while the CLI binary and import name remain `promptdiff`.

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

## 🩺 Environment Doctor

Diagnose local environment readiness, LLM API keys, optional packages (`tiktoken`, `sentence-transformers`, `mlflow`, `wandb`), and disk cache engine with a single command:

```bash
promptdiff doctor
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

### 🤖 Standalone Pull Request Commenter (`scripts/pr_commenter.py`)

For non-composite CI environments (Jenkins, GitLab CI, Buildkite, or custom GitHub Actions steps), use the standalone PR commenting script:

```bash
# Run regression evaluation exporting report JSON
promptdiff test prompts/system_v1.txt prompts/system_v2.txt \
  --inputs datasets/testcases.jsonl \
  --mock \
  --export-json report.json

# Post or update sticky Markdown evaluation comment on PR
python scripts/pr_commenter.py \
  --report report.json \
  --repo "$GITHUB_REPOSITORY" \
  --pr "$PR_NUMBER" \
  --token "$GITHUB_TOKEN"
```

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

## 🐍 Python SDK

Use `promptdiff` programmatically inside Python applications or evaluation scripts:

```python
import promptdiff
from promptdiff.core.models import TestCase

# Run regression evaluation
report = promptdiff.compare(
    v1="prompts/support_v1.txt",
    v2="prompts/support_v2.txt",
    dataset=[
        TestCase(id="tc1", vars={"query": "Reset password"}),
        TestCase(id="tc2", vars={"query": "Billing question"}),
    ],
    model="gpt-4o",
    mock=True,
    assertions=["cost_delta <= 15%", "latency_delta <= 20%"],
)

print(f"Passed: {report.verdict.passed}")
print(f"Cost Delta: {report.verdict.cost_delta_pct:.1f}%")

# Compress prompt tokens while maintaining quality
shrunk = promptdiff.shrink(
    prompt="Please kindly act as an AI and answer: {{query}}",
    dataset=[TestCase(id="1", vars={"query": "Help"})],
    mock=True,
)
print(f"Compressed Prompt: {shrunk.compressed_prompt}")
```

---

## 📦 Installation & Modular Extras

PromptDiff is built with a slim, featherweight core and modular extras so you only install what you need:

```bash
# Core CLI & CI runner (typer, rich, pydantic, httpx, jinja2, pyyaml, tenacity, numpy)
pip install promptdiff-eval

# Semantic dense embedding similarity (sentence-transformers)
pip install "promptdiff-eval[semantic]"

# Interactive split-screen Terminal UI (Textual)
pip install "promptdiff-eval[tui]"

# Streamlit telemetry web dashboard
pip install "promptdiff-eval[ui]"

# All optional components
pip install "promptdiff-eval[all]"
```

---

## 🧩 Advanced & Extended Modules

| Command / Tool | Extra Required | Description |
| :--- | :---: | :--- |
| `promptdiff cache-impact` | *Core* | KV-cache prefix breakpoint analyzer & monthly financial cash loss forecaster. |
| `promptdiff replay-traces` | *Core* | Production OpenTelemetry & Langfuse shadow replayer with automated PII masking. |
| `promptdiff arena` | *Core* | Evaluate $N$ prompt versions with Bayesian Bradley-Terry & ELO skill ratings. |
| `promptdiff studio` | *Core* | Launch zero-dependency local-first visual diff web studio & playground. |
| `promptdiff mcts` | *Core* | Active Monte Carlo Tree Search prompt optimizer with Pareto frontier. |
| `promptdiff redteam` | *Core* | Multi-turn TAP adversarial red-teaming (steganography & CVSS risk matrix). |
| `promptdiff cascade` | *Core* | Confidence-aware model cascade router & enterprise ROI forecaster. |
| `promptdiff check` | *Core* | Static linting & token cost analysis for prompt templates. |
| `promptdiff serve` | *Core* | Launch FastAPI REST API server & playground (`pip install fastapi uvicorn`). |
| `promptdiff diff` | *Core* | Instant side-by-side terminal syntax diff without calling model APIs. |
| `promptdiff pricing` | *Core* | Query token pricing and cost calculations for 30+ providers. |
| `promptdiff fuzz` | *Core* | Red-teaming security fuzzer scanning 20 distinct adversarial injection vectors. |
| `promptdiff tui` | `[tui]` | Launch interactive split-screen terminal workspace (`pip install promptdiff-eval[tui]`). |
| `promptdiff ui` | `[ui]` | Launch Streamlit web dashboard for interactive telemetry (`pip install promptdiff-eval[ui]`). |
| `promptdiff optimize` | *Core* | Reflective auto-prompt optimizer (DSPy style) using meta-model feedback. |
| `promptdiff shrink` | *Core* | Token compressor pruning boilerplate fluff while preserving 100% output quality. |
| `promptdiff cache-sim` | *Core* | Prefix caching hit rate analyzer and ROI forecaster. |
| `promptdiff history` | *Core* | Benchmark prompt quality and cost evolution across Git revisions. |

---

## 🛡️ Production Engineering Standards & Quality Assurance

PromptDiff is built to enterprise MLOps standards with zero tolerance for unverified code or silent regressions:

| Dimension | Quality Standard | Verification |
| :--- | :--- | :--- |
| **Comprehensive Test Suite** | 390+ unit, integration, and security tests (see CI badge above for live count) | `pytest` passing on Linux, macOS, and Windows |
| **Test Coverage** | 92%+ branch & statement coverage (see CI badge above) | Automated threshold enforcement in CI (`--cov-fail-under=85`) |
| **Isolated Code Sandbox** | Subprocess execution with resource limits (`RLIMIT_AS`, `RLIMIT_CPU`) | Exploit-tested AST/memory barriers & strict timeout handling |
| **Strict Type Safety** | 100% type-annotated codebase (PEP 561 compliant `py.typed`) | `mypy --strict promptdiff` (0 errors across 121 source files) |
| **Code Formatting & Linting** | Automated style checking & import order | `ruff check .` & `ruff format --check .` in pre-commit |
| **Cryptographic Provenance** | HMAC-SHA256 zero-width prompt steganography | Constant-time tamper detection (`hmac.compare_digest`) |
| **Schema Drift Protection** | Automated drift protection against JSON schema divergence | `DiffReport.model_json_schema()` verified in CI pipeline |

---

## 🏛️ Architecture Deep-Dive

Detailed system design documentation, architectural diagrams, mathematical formulations, and core engineering decisions are available in:

👉 [**Technical Architecture & System Design Deep-Dive (PORTFOLIO.md)**](PORTFOLIO.md)

---

## 🔒 Data Privacy & Local Storage Disclosure

PromptDiff operates under an absolute **local-first, zero-telemetry exfiltration** guarantee:
- **Local Persistence Only:** Evaluation runs and token metrics are written to local SQLite storage (`.promptdiff/telemetry.db`). No prompt contents, outputs, or traces are ever sent to external cloud servers.
- **Automated Retention Management:** Automatically delete historical records older than $N$ days with `--db-retention-days <N>` or run `promptdiff db prune --days 14`.
- **Ephemeral Storage:** Run with `--db-path ":memory:"` for zero disk persistence.
- Complete security documentation and disclosure SLAs are available in [SECURITY.md](SECURITY.md).

---

## 🤝 Community & Contributing

- [Contributing Guide](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security Policy](SECURITY.md)
- [Changelog](CHANGELOG.md)

---

## 📈 Star History

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=latryee/promptdiff&type=Date)](https://star-history.com/#latryee/promptdiff&Date)

</div>

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
