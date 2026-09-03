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
