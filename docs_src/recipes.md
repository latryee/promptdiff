# Curated Recipe Catalog

PromptDiff includes pre-built domain starter kits to kickstart prompt regression testing with tailored datasets, schemas, and evaluators.

---

## Available Recipes

| Recipe Name | Focus Area | Evaluators Included | Command |
| :--- | :--- | :--- | :--- |
| **`rag-qa`** | Retrieval-Augmented Generation | Faithfulness, context recall, latency | `promptdiff recipe pull rag-qa` |
| **`json-extractor`** | Strict Structured Outputs | JSON validity, schema compliance, AST drift | `promptdiff recipe pull json-extractor` |
| **`sql-gen`** | Natural Language to SQL | SQL syntax validity, table schema adherence | `promptdiff recipe pull sql-gen` |
| **`security-guard`** | Adversarial Defense | Injection resistance, system prompt leakage | `promptdiff recipe pull security-guard` |

---

## Using Recipes

### 1. List Available Recipes

```bash
promptdiff recipe list
```

### 2. Pull a Recipe into Your Project

```bash
promptdiff recipe pull rag-qa --dest my-rag-evals
```

This scaffolds:
- `prompts/system_v1.txt`: Baseline system prompt with context grounding rules.
- `prompts/system_v2.txt`: Candidate optimized system prompt.
- `testcases.jsonl`: Curated evaluation cases including adversarial trap questions.
- `promptdiff.yaml`: Pre-configured evaluator thresholds.

### 3. Run the Recipe Suite

```bash
cd my-rag-evals
promptdiff test prompts/system_v1.txt prompts/system_v2.txt --inputs testcases.jsonl --mock
```
