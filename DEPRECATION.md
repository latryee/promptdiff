# PromptDiff Deprecation & Semantic Versioning Policy

This document establishes the official API stability guarantees, breaking change management process, and deprecation lifecycle for **PromptDiff** (SDK, CLI, REST API, and data storage formats).

---

## 📌 Semantic Versioning Principles

PromptDiff strictly adheres to [Semantic Versioning 2.0.0](https://semver.org/):

$$\text{Version} = \text{MAJOR}.\text{MINOR}.\text{PATCH}$$

- **`MAJOR` (e.g., 3.0.0 -> 4.0.0)**:
  - Incompatible breaking changes to public Python SDK interfaces (`promptdiff.sdk`).
  - Removal of CLI subcommands, renaming of required flags, or fundamental syntax changes.
  - Breaking changes to the core JSON report format (`DiffReport` schema).
  - Breaking changes to REST API contracts (`/api/v1/*`).
- **`MINOR` (e.g., 3.4.0 -> 3.5.0)**:
  - New backward-compatible features, evaluators, provider adapters, or metrics.
  - New optional CLI commands and flags.
  - Additive extensions to `DiffReport` that do not invalidate existing consumers.
  - Deprecation warnings signaling planned future breaking changes.
- **`PATCH` (e.g., 3.4.0 -> 3.4.1)**:
  - Backward-compatible bug fixes, security hardening, and performance optimizations.
  - Documentation, typing refinements, and packaging improvements.

---

## ⏳ Deprecation Lifecycle

To ensure production stability for enterprise CI/CD pipelines, PromptDiff follows a three-stage deprecation process before any public interface is retired:

```mermaid
flowchart LR
    A["Active Stable API"] --> B["Stage 1: Deprecation Notice"]
    B --> C["Stage 2: Grace Period (>= 1 Minor Cycle)"]
    C --> D["Stage 3: Removal in Next Major Release"]
```

### Stage 1: Deprecation Notice
When an API element or CLI option is designated for deprecation:
1. **Code Annotations**: The Python function or parameter is decorated with a runtime `DeprecationWarning` (using standard library `warnings.warn(..., category=DeprecationWarning, stacklevel=2)`).
2. **Documentation**: Clear `@deprecated` notices are added to docstrings, the [CHANGELOG.md](CHANGELOG.md), and online reference docs with recommended replacement alternatives.
3. **CLI Warning**: If a CLI option is deprecated, invoking it emits a non-fatal warning on `stderr` indicating the replacement option.

### Stage 2: Grace Period
- Deprecated APIs remain **100% operational and fully tested** throughout the grace period.
- The grace period spans **at least one minor release cycle (e.g., v3.4 to v4.0)** or a minimum of **90 calendar days**.
- No breaking behavior modifications are introduced during this phase.

### Stage 3: Removal
- The deprecated interface is removed only during the next **MAJOR** release (e.g., `4.0.0`).
- The release notes will provide an explicit migration guide and automated transformation recipes where applicable.

---

## 🛡️ Public vs. Private API Surface

| Component | Scope | Stability Guarantee |
| :--- | :--- | :--- |
| **`promptdiff.sdk`** | Public | Guaranteed stable across MINOR and PATCH releases. |
| **Core Models (`DiffReport`, `TestCase`)** | Public | Strict schema versioning (`schema_version` tag); additive only. |
| **CLI Commands (`promptdiff test`, etc.)** | Public | Command names and exit codes are guaranteed stable. |
| **REST API (`/api/v1/*`)** | Public | Versioned endpoint pathing; breaking changes require `/api/v2/*`. |
| **Internal Modules (`promptdiff._*`, `promptdiff.core._*`)** | Private | Internal implementation detail; may change without notice. |

---

## 🔄 Schema Migration Guarantees

Any serialized output format (such as `DiffReport` JSON schemas or SQLite `.promptdiff/telemetry.db` tables) includes an explicit schema version identifier. 

- Newer versions of PromptDiff will maintain backward compatibility with previous schema versions.
- Automated migrations or schema upgrades are performed non-destructively.
