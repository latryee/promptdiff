# PromptDiff Dogfooding & Self-Test Suite

This directory contains internal self-test assets used for dogfooding PromptDiff on its own prompt evolution:
- prompts/system_v1.txt: Baseline support prompt.
- prompts/system_v2.txt: Hand-crafted concise candidate prompt.
- prompts/system_v3_optimized.txt: DSPy auto-optimizer tuned prompt (promptdiff optimize).
- prompts/system_shrunk.txt: Token compressor shrunk prompt (promptdiff shrink).
- testcases.jsonl: Evaluation test dataset.
- promptdiff.yaml: Evaluation configuration and regression assertions.

### Running Self-Test

```bash
promptdiff test .promptdiff-self-test/prompts/system_v1.txt .promptdiff-self-test/prompts/system_v2.txt --inputs .promptdiff-self-test/testcases.jsonl --mock
```
