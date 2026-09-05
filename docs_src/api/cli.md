# CLI Command Reference

`promptdiff` provides an enterprise-grade command-line interface built on Typer and Rich.

---

## `promptdiff test` / `promptdiff run`

Runs regression comparison between two prompt versions across a dataset.

```bash
promptdiff test prompts/v1.txt prompts/v2.txt \
  --inputs datasets/testcases.jsonl \
  --model gpt-4o \
  --eval json_validity,latency,cost,similarity,llm_judge \
  --assert "cost_delta <= 10%, latency_delta <= 15%" \
  --db-retention-days 30 \
  --fail-on-regression \
  --export-html report.html \
  --export-markdown report.md \
  --export-json report.json
```

### Key Options

| Option | Flag | Description |
| :--- | :--- | :--- |
| `--inputs` | `-i` | Path to dataset file (`.jsonl`, `.yaml`, `.csv`, `.json`). |
| `--model` | `-m` | Target LLM model name (e.g. `gpt-4o`, `claude-3-5-sonnet`). Default: `gpt-4o`. |
| `--eval` | `-e` | Comma-separated list of evaluators to execute (`json_validity`, `cost`, `latency`, `similarity`, `faithfulness`, `security`). |
| `--assert` | `-a` | Regression assertion thresholds (e.g. `"cost_delta <= 10%, similarity >= 0.8"`). |
| `--mock` | | Execute using deterministic offline mock provider (no external API calls or API keys required). |
| `--db-retention-days` | | Automatically prune SQLite historical telemetry database runs older than $N$ days. |
| `--estimate` | | Pre-calculate and display token and financial estimates before running evaluations. |
| `--fail-on-regression` | | Exit with return code `1` if any assertion threshold is breached (ideal for CI/CD gates). |
| `--export-html` | | Write standalone zero-dependency interactive HTML diff report. |
| `--export-markdown` | | Output GitHub-flavored markdown report table. |
| `--export-json` | | Output machine-readable JSON schema report. |

---

## `promptdiff doctor`

Diagnoses your local runtime environment, SQLite cache engine, and LLM API provider keys:

```bash
promptdiff doctor
```

Outputs connectivity checks for:
- OpenAI API (`OPENAI_API_KEY`)
- Anthropic API (`ANTHROPIC_API_KEY`)
- Google Gemini API (`GEMINI_API_KEY`)
- Local disk cache integrity & SQLite WAL mode
- Optional acceleration packages (`tiktoken`, `sentence-transformers`, `streamlit`, `textual`)

---

## `promptdiff arena`

Runs multi-model A/B/C/D evaluation benchmarks across prompt candidates using Bayesian Bradley-Terry & ELO skill ratings:

```bash
promptdiff arena \
  --prompts prompts/v1.txt,prompts/v2.txt \
  --models gpt-4o,claude-3-5-sonnet,gemini-2.0-flash \
  --inputs datasets/testcases.jsonl \
  --mock
```

---

## `promptdiff shrink`

Prunes redundant boilerplate fluff and compresses prompts while maintaining 100% output quality:

```bash
promptdiff shrink prompts/verbose.txt --inputs testcases.jsonl --target-reduction 0.30
```

---

## `promptdiff fuzz`

Adversarial red-teaming security fuzzer scanning 20 distinct prompt injection, jailbreak, and extraction attack vectors:

```bash
promptdiff fuzz prompts/system_v1.txt --model gpt-4o --mock
```

---

## `promptdiff cache-sim` & `promptdiff cache-impact`

Analyzes prefix caching hit rate and models monthly enterprise financial ROI:

```bash
# Prefix cache simulator
promptdiff cache-sim prompts/system_v1.txt --inputs testcases.jsonl

# KV-cache breakpoint analyzer & financial loss forecaster
promptdiff cache-impact prompts/v1.txt prompts/v2.txt --monthly-requests 1000000
```

---

## `promptdiff mcts`

Active Monte Carlo Tree Search prompt optimizer with Pareto frontier exploration:

```bash
promptdiff mcts prompts/system.txt --inputs testcases.jsonl --budget 50 --mock
```

---

## `promptdiff db`

Query and maintain persistent SQLite evaluation telemetry:

```bash
# View recent evaluation runs
promptdiff db stats

# Identify test cases with the highest regression failure frequency
promptdiff db hotspots

# Prune runs older than 14 days
promptdiff db prune --days 14
```

---

## `promptdiff pricing`

Query local token pricing registry and calculate precise token costs for 30+ providers:

```bash
promptdiff pricing gpt-4o
promptdiff pricing claude-3-5-sonnet
```

---

## `promptdiff check`

Static linting, token count estimation, and variable validation for prompt templates:

```bash
promptdiff check prompts/system_v1.txt
```

---

## `promptdiff studio` & `promptdiff ui`

Launch interactive visual tools:

```bash
# Zero-dependency local web studio
promptdiff studio

# Streamlit interactive telemetry dashboard (requires pip install "promptdiff-eval[ui]")
promptdiff ui
```
