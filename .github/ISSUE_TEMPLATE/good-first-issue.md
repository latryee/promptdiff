---
name: Good First Issue
about: A bite-sized task suitable for newcomers and first-time contributors
title: "[Good First Issue]: "
labels: ["good first issue", "help wanted"]
assignees: ''
---

### 🎯 Overview & Problem Description
<!-- Provide a clear, self-contained description of what needs to be added, fixed, or improved. -->

### 📁 Relevant Files & Code Locations
<!-- Link to specific files and functions to give newcomers an immediate starting point. -->
- `promptdiff/...`
- `tests/...`

### 📋 Acceptance Criteria
<!-- What does a successful implementation require? -->
- [ ] Implementation complete and aligned with existing code conventions.
- [ ] Unit tests added covering both happy and edge cases.
- [ ] `pytest -q` passes without regressions.
- [ ] `ruff check .` and `ruff format --check .` pass.
- [ ] `mypy promptdiff` reports zero type errors.

### 💡 Suggested Approach & Mentorship Tips
<!-- Step-by-step guidance to help the contributor implement the solution smoothly. -->
1. Fork the repo and set up your virtual environment (`pip install -e ".[dev]"`).
2. Create a new branch: `git checkout -b feat/your-feature-name`.
3. Make your changes and add corresponding tests in `tests/`.
4. Run `pytest -q ; ruff check . ; mypy promptdiff` to verify everything is green.
5. Submit your PR and tag `@latryee` for review!
