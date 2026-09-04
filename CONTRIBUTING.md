# Contributing to PromptDiff

Thank you for your interest in contributing to **PromptDiff**! ⚡

Whether you are reporting a bug, proposing a new feature, writing documentation, or submitting a pull request, we welcome your contributions.

---

## 🛠️ Development Setup

PromptDiff requires **Python 3.10+**. We recommend using a virtual environment:

```bash
# 1. Fork and clone the repository
git clone https://github.com/latryee/promptdiff.git
cd promptdiff

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install in editable mode with development dependencies
pip install --upgrade pip
pip install -e ".[dev,all]"
```

---

## 🌱 Good First Issues & Beginner Onboarding Tasks

New to PromptDiff or looking for an easy place to start? We actively curate bite-sized, accessible tasks labeled [`good first issue`](https://github.com/latryee/promptdiff/labels/good%20first%20issue). Here are 5 great entry points:

1. **Add a New LLM Pricing Model Entry** (`promptdiff/core/pricing.py`):
   - Add input/output token pricing per 1M tokens for newly released models (e.g. `gemini-2.0-flash`, `claude-3-7-sonnet`, `deepseek-v3`).
   - Add matching assertions in `tests/core/test_providers.py`.
2. **Implement a Specialized Evaluator Metric** (`promptdiff/evaluators/`):
   - Create a lightweight evaluator checking output formatting rules (e.g., Markdown table validator, YAML schema verifier, or reading grade level scorer).
   - Inherit from `BaseEvaluator` and register it in `EvaluatorRegistry`.
3. **Expand the Adversarial Red-Teaming Fuzzer** (`promptdiff/security/fuzzer.py`):
   - Add new jailbreak, indirect prompt injection, or system prompt exfiltration payload patterns to increase test coverage.
4. **CLI Output Polish & Terminal UX** (`promptdiff/cli/`):
   - Improve Rich table formatting, add colored status badges, or enhance error hints when user datasets contain missing fields.
5. **Add End-to-End Evaluation Recipes** (`examples/recipes/`):
   - Contribute a runnable evaluation recipe demonstrating prompt regression testing with frameworks like LangGraph, Instructor, or LiteLLM.

---

## 🧪 Testing & Code Quality

Before submitting a pull request, ensure that all tests, linters, and type checks pass:

### 1. Run the Test Suite
```bash
pytest -v
```

### 2. Check Code Style & Linting
We use **Ruff** for high-speed linting and formatting:
```bash
# Lint checks
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Format checks
ruff format --check .

# Auto-format
ruff format .
```

### 3. Strict Type Checking
We maintain strict **Mypy** type safety:
```bash
mypy promptdiff
```

---

## 🚀 Creating a Pull Request

1. **Create a branch** for your work:
   ```bash
   git checkout -b feat/my-new-feature
   # or
   git checkout -b fix/my-bugfix
   ```

2. **Commit your changes**:
   We follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat(evaluator): add semantic perplexity scorer`
   - `fix(cli): resolve windows terminal encoding bug`
   - `docs: improve pytest plugin quickstart example`
   - `test: add unit test for council evaluator consensus`

3. **Verify locally**:
   Ensure `pytest`, `ruff check`, and `mypy` all pass cleanly.

4. **Submit your PR**:
   Open a pull request against `main`. Ensure your PR description explains the motivation, implementation details, and verification steps.

---

## 💬 Community & Support

- Questions & Discussions: [GitHub Discussions](https://github.com/latryee/promptdiff/discussions)
- Bug Reports & Feature Requests: [GitHub Issues](https://github.com/latryee/promptdiff/issues)
