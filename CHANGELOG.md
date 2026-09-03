# Changelog

All notable changes to **PromptDiff** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.4.0] - 2026-09-03

### Added
- **Council of Judges Evaluator**: Swarm-consensus multi-model evaluation (`CouncilOfJudgesEvaluator`) aggregating verdicts across diverse LLM judges.
- **Streaming TTFT Profiler**: Real-time profiling for Time-To-First-Token (TTFT), token inter-arrival jitter, and throughput distributions.
- **Semantic Drift Watch Daemon**: Automated drift watcher monitoring prompt outputs over time.
- **Prompt Watermarking**: Cryptographic watermark injection and verification for LLM prompt provenance.
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
