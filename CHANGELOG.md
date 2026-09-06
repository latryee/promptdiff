# Changelog

All notable changes to **PromptDiff** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### 📋 Breaking Change & Deprecation Policy
PromptDiff enforces strict Semantic Versioning (`MAJOR.MINOR.PATCH`):
- **Major Releases (`X.0.0`)**: Permitted to introduce breaking changes in the public Python SDK (`promptdiff.sdk`), CLI subcommands, and report schemas. All breaking modifications require a deprecation notice in at least one preceding minor release.
- **Minor Releases (`0.Y.0`)**: Backward-compatible new features, metrics, evaluators, and performance enhancements. May introduce deprecation warnings.
- **Patch Releases (`0.0.Z`)**: Backward-compatible bug fixes and security hardening.
- Detailed migration timelines and API stability scopes are documented in [DEPRECATION.md](DEPRECATION.md).

---

## [3.5.0] - 2026-09-06

### Enterprise Architecture & Reliability
- **Standardized CI Exit Codes**: Introduced `ExitCode` enum with standardized exit statuses: `0` (Success), `1` (Regression Detected), `2` (Configuration / Dataset Syntax Error), `3` (Provider API / Auth / Rate Limit Error), and `4` (Internal System Error).
- **Streaming Dataset Engine & Dynamic Filtering**: Added lazy file streaming via `stream_dataset()` for arbitrary scale JSONL/YAML/CSV datasets with negligible memory footprint. Added `--tags` filtering and `--limit` test-case truncation alongside line-numbered `DatasetError` diagnostics.
- **Security & PII Redaction Engine**: Implemented `promptdiff.security.redaction` with automated regex masking for LLM provider API keys (OpenAI, Anthropic, Gemini, AWS, Hugging Face), Bearer/JWT tokens, and PII (emails, SSNs, credit cards). Integrated via `--redact` CLI flag, SDK parameters, and `SecretRedactingFilter` in logging.
- **Runtime Execution Provenance**: Enriched `DiffReport` and `RunResult` with full execution provenance (`RunProvenance`), capturing Git commit SHA, branch, dirty status, machine architecture, OS, Python/PromptDiff versions, dataset SHA-256 hash, and configuration snapshots.
- **SQLite Cache Resilience & Self-Healing**: Hardened `DiskCache` with schema v2 (`cache_meta`), SQLite WAL mode, 30s busy timeouts, self-healing decode error eviction, automatic corruption quarantine (`cache.sqlite.corrupt.<timestamp>`), and bounded TTL pruning.
- **Database Schema Migrations**: Introduced formal `schema_migrations` tracking in SQLite telemetry database with automated migration to v2, persisting experiment IDs, baseline IDs, model parameters, and evaluator lists.
- **Exact Match Evaluator**: Added deterministic `ExactMatchEvaluator` supporting case-folding, whitespace collapsing, punctuation normalization, and unified diff output generation. Added dynamic evaluator class loading via `"module.path:ClassName"`.
- **Advanced Pricing & Reasoning Token Support**: Upgraded pricing registry to `2025.03` rates with support for cached input tokens, prompt reasoning tokens (o1, o3-mini, DeepSeek-R1), and runtime custom pricing overrides (`register_model_price`).
- **Pytest Plugin Modernization**: Added declarative `@pytest.mark.promptdiff` test marker, `report` and `diff_report` fixtures, and automated post-test regression assertion enforcement.
- **Provider Architecture & Capabilities**: Added `ProviderCapabilities` introspection (`supports_streaming`, `supports_system_prompt`, `supports_tools`, `supports_json_schema`) and unified provider exception classification (`classify_provider_exception`).

### Security & Packaging
- **PyPI Package Renamed to `promptdiff-eval`**: Official distribution package renamed to `promptdiff-eval` to secure reliable installation via `pip install promptdiff-eval`.
- **FastAPI Server Security Hardening**: Added API key authentication (`--api-key` / `PROMPTDIFF_API_KEY`), sliding-window IP rate limiting (100 req/min), configurable CORS policies, and safe-by-default localhost binding (`127.0.0.1`).
- **Composite GitHub Action Shell Injection Neutralized**: Replaced unquoted template parameters in `action.yml` with environment-variable-backed arguments to eliminate command injection vectors.

### Fixed & Enhanced
- **JS AST Diff Bug Fixed**: Corrected token boundary calculation and syntax node diffing in JavaScript AST evaluator.
- **Verdict & Status Consistency Harmonization**: Harmonized regression verdict computation across CLI, reporters, and SDK so test status codes and summary badges strictly align.
- **Internal Dogfooding Suite**: Established `.promptdiff-self-test/` containing prompt evolution assets (`system_v1` -> `system_v2` -> `system_v3_optimized` / `system_shrunk`) and added automated self-testing step in CI.
- **Cross-Platform Normalization**: Added `.gitattributes` enforcing consistent LF line endings and binary file handling across Windows, macOS, and Linux runners.

---

## [3.4.1] - 2026-09-06

### Changed
- Relocated root prompt assets to `.promptdiff-self-test/` for internal dogfooding and prompt evolution verification.

---

## [3.4.0] - 2026-09-03

### Added
- **Council of Judges Evaluator**: Swarm-consensus multi-model evaluation (`CouncilOfJudgesEvaluator`) aggregating verdicts across diverse LLM judges.
- **Streaming TTFT Profiler**: Real-time profiling for Time-To-First-Token (TTFT), token inter-arrival jitter, and throughput distributions.
- **Semantic Drift Watch Daemon**: Automated drift watcher monitoring prompt outputs over time.
- **Prompt Watermarking**: Cryptographic watermark injection (HMAC-SHA256 zero-width steganography) and constant-time verification for LLM prompt provenance and tamper detection.
- **Edge Quantization Parity**: Benchmark quantization drift (FP16 vs INT8 vs INT4) on locally served models.
- **Property-Based Invariant Testing**: Automated hypothesis-style invariant fuzzing for prompt output constraints.
- **EU AI Act Compliance Auditor**: Static & behavioral risk categorization adhering to EU AI Act transparency rules.
- **Reflection Loop Benchmark**: Benchmark agentic self-correction iterations and convergence rates.
- **Jupyter Notebook Exporter**: Export regression comparison results directly to rich Jupyter notebooks (`.ipynb`).
- **Prompt JIT Compiler**: Template optimizer pre-compiling static prompt tokens for optimal prefix cache boundaries.
- **CLI Commands**: Added `promptdiff serve` (FastAPI REST API server) and `promptdiff check` (prompt static linting & cost analysis).
- **Pytest Plugin**: Added `prompt_diff` fixture with both `await prompt_diff.compare(...)` and `prompt_diff.sync_compare(...)` alongside `promptdiff_eval`.
- **PEP 561 Typing**: Shipped `py.typed` marker for complete inline IDE and type-checker support.
- **Modern Model Pricing**: Added `claude-3-7-sonnet`, `gpt-4.5`, `gemini-2.5-pro`, and `o3-mini` to pricing registry.

---

## [3.3.0] - 2026-08-25

### Added
- **Shadow Traffic Replay**: Asynchronous replay of real production request logs through candidate prompts.
- **Model Cascading Router**: Cost-latency router falling back to smaller models when confidence thresholds pass.
- **Canary Rollout Planner**: Canary deployment configuration generator (Kubernetes / Istio / Envoy).
- **SLA Simulator**: P95/P99 latency stress simulation under simulated production concurrency.
- **AI Fairness Evaluator**: Counterfactual demographic perturbation tester for demographic neutrality.
- **Citation Pointer Evaluator**: Hallucination detection pointing sentences directly back to source context.
- **Needle In A Haystack Tester**: Deep context retrieval accuracy evaluator at varying context depths.
- **Personas Stress Generator**: Automated generation of extreme customer personas for prompt edge testing.
- **Dynamic Exemplars**: In-context learning few-shot selector ranking exemplars by similarity.
- **Schema Auto-Repair**: Resilient JSON parsing and AST structural validator.
- **Saliency Mapper**: Token attribution and importance weight mapping.
- **Distillation Engine**: Dataset exporter generating fine-tuning pairs for small model distillation.
- **LSP Bridge**: Language Server Protocol diagnostic bridge for VS Code and Cursor prompt editing.

---

## [3.2.0] - 2026-08-15

### Added
- **Adversarial Fuzzer**: Red-teaming fuzzer probing 20 distinct jailbreak vectors (roleplay, encoding, injection).
- **Prefix Caching Simulator**: Prefix cache boundary optimizer predicting hit rates and dollar savings.
- **Dataset Mutator**: Seed dataset expander generating synthetic typographical and semantic variations.
- **Git History Tracker**: Prompt regression benchmark tracking performance across git revisions.
- **Statistical Significance Engine**: Bootstrap hypothesis testing and p-value calculation for score deltas.
- **Self-Contained Single-File HTML Bundle**: Zero-dependency standalone report exporter.

---

## [3.1.0] - 2026-08-01

### Added
- **Pytest Plugin**: Official pytest integration via `pytest-promptdiff`.
- **Prompt Token Shrinker**: Heuristic and model-driven prompt compressor reducing tokens without quality loss.
- **Agent Trajectory Evaluator**: Multi-step tool use, function calling, and action trajectory regression evaluator.
- **OpenTelemetry & Langfuse Export**: Automated telemetry export to OpenTelemetry collectors and Langfuse.
- **Python SDK**: High-level synchronous and asynchronous Python SDK (`import promptdiff`).

---

## [3.0.0] - 2026-07-15

### Added
- **Initial Enterprise Release**: Core async runner, Typer CLI, Rich terminal dashboard, deterministic SQLite caching, model pricing registry, and standalone HTML reports.
