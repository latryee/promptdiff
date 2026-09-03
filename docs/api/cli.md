# CLI Command Reference

`promptdiff` provides an enterprise-grade command-line interface built on Typer and Rich.

## `promptdiff test` / `promptdiff run`
Runs regression comparison between two prompt versions across a dataset.

```bash
promptdiff test prompts/v1.txt prompts/v2.txt \
  --inputs datasets/testcases.jsonl \
  --model gpt-4o \
  --eval json_validity,latency,cost,similarity,llm_judge \
  --assert "cost_delta <= 10%, latency_delta <= 15%" \
  --db-retention-days 30 \
  --fail-on-regression
```

### Key Options
- `--inputs, -i`: Path to dataset file (`.jsonl`, `.yaml`, `.csv`, `.json`).
- `--model, -m`: Target LLM model name (default: `gpt-4o`).
- `--eval, -e`: Comma-separated list of evaluators to execute.
- `--assert, -a`: Regression assertion threshold.
- `--mock`: Run with deterministic offline mock provider (no API keys needed).
- `--db-retention-days`: Automatically prune historical telemetry database runs older than N days.
- `--estimate`: Show pre-execution token/cost estimation before running.

---

## `promptdiff doctor`
Diagnoses your local runtime, SQLite cache engine, and LLM API provider keys.

```bash
promptdiff doctor
```

---

## `promptdiff arena`
Runs multi-model A/B/C/D evaluation benchmarks across prompt candidates.

```bash
promptdiff arena \
  --prompts prompts/v1.txt,prompts/v2.txt \
  --models gpt-4o,claude-3-5-sonnet,gemini-2.0-flash \
  --inputs datasets/testcases.jsonl \
  --mock
```

---

## `promptdiff db`
Query and maintain persistent SQLite evaluation telemetry.

- `promptdiff db stats`: View recent chronological evaluation runs.
- `promptdiff db hotspots`: Identify test cases with the highest regression failure frequency.
- `promptdiff db prune --days 30`: Prune runs older than 30 days.

---

## `promptdiff shrink`
Prunes redundant instructions while guaranteeing zero quality regression.

```bash
promptdiff shrink prompts/verbose.txt --inputs testcases.jsonl --target-reduction 0.30
```
