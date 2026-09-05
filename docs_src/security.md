# Security & Sandboxing

PromptDiff enforces rigorous security standards to prevent code injection, data exfiltration, and unauthorized network access during automated evaluations.

---

## 1. Hardened Subprocess Code Sandbox

When executing user-defined evaluators or custom code assertions:
- **Process Isolation**: Code runs inside an isolated OS subprocess invoked with isolated flags (`python -I -s -B`).
- **Resource Constraints**: Strict limits enforced via POSIX `setrlimit` (`RLIMIT_AS` memory limits and `RLIMIT_CPU` processor time caps).
- **Filesystem Barriers**: Disables arbitrary file system writes outside designated ephemeral scratch directories.
- **Strict Timeouts**: Subprocesses exceeding evaluation deadlines are terminated immediately via `SIGKILL` after grace periods.

---

## 2. Watermarking & Steganography

PromptDiff includes cryptographic prompt provenance:
- **Zero-Width Steganography**: Embeds HMAC-SHA256 signatures into prompt templates using invisible zero-width Unicode characters.
- **Tamper Verification**: Validates prompt origin and detects unauthorized intermediate mutations using constant-time string comparison (`hmac.compare_digest`).

---

## 3. Data Sovereignty & Zero Cloud Telemetry

PromptDiff operates under an absolute **local-first** policy:
- Evaluation runs, logs, and token metrics remain strictly on local disk in SQLite (`.promptdiff/telemetry.db`).
- Zero prompt contents, LLM responses, or user inputs are transmitted to external servers.
- Use `--db-path ":memory:"` for completely ephemeral in-memory evaluation runs with zero disk writes.
- To prune telemetry data: `promptdiff db prune --days 14` or configure `--db-retention-days <N>`.
