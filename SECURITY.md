# Security Policy

## Supported Versions

PromptDiff provides security updates for the following release tracks:

| Version | Supported          |
| ------- | ------------------ |
| 3.4.x   | :white_check_mark: |
| 3.x.x   | :white_check_mark: |
| < 3.0.0 | :x:                |

---

## Reporting a Vulnerability

We take the security of PromptDiff seriously. If you discover a vulnerability or security concern, please follow responsible disclosure guidelines:

1. **Do NOT report vulnerabilities through public GitHub issues.**
2. Report vulnerabilities via GitHub's private vulnerability reporting feature:
   - Navigate to the **Security** tab of the repository
   - Click **Report a vulnerability**
3. Or email the maintainers directly at: `security@promptdiff.org`

### What to Include
- Detailed description of the issue
- Steps or a minimal reproducible example to reproduce the vulnerability
- Potential impact and severity assessment
- Any suggested fixes or mitigations

### Response Timeline & Triage SLAs

PromptDiff maintains explicit Service Level Agreements (SLAs) for vulnerability triage, remediation, and patch distribution:

| Severity | CVSS v3.1 Range | Initial Acknowledgment | Root-Cause Confirmation | Security Patch Release |
| :--- | :---: | :---: | :---: | :---: |
| **Critical** | 9.0 – 10.0 | **< 24 hours** | **< 48 hours** | **< 72 hours** |
| **High** | 7.0 – 8.9 | **< 48 hours** | **< 4 business days** | **< 7 business days** |
| **Medium** | 4.0 – 6.9 | **< 5 business days** | **< 10 business days** | Next minor release |
| **Low** | 0.1 – 3.9 | **< 10 business days** | Best effort | Next minor release |

- **Emergency Hotfixes:** Critical vulnerabilities (e.g. sandbox escapes, credential leakage, remote code execution) receive expedited emergency hotfixes released to PyPI outside the standard release cycle.
- **CVE Attribution:** Reporters of validated security issues will be credited in GitHub Security Advisories and release notes.

---

## 🔒 Local Data Storage & Privacy Policy

PromptDiff operates under an absolute **local-first, zero-telemetry exfiltration** architecture.

### What Data Is Stored Locally
When running evaluations with telemetry enabled, PromptDiff records evaluation records into a local SQLite database (default: `.promptdiff/telemetry.db`):
- **Run Metadata**: Unique `run_id`, timestamp, prompt version identifiers, overall pass/fail status, and aggregate cost/latency deltas.
- **Execution Telemetry**: Rendered prompt inputs, model completion outputs, token usage counts, and evaluator scores for each test case.

### Privacy Guarantees
- **Zero Cloud Exfiltration**: PromptDiff does **NOT** collect, transmit, or phone home any prompt contents, model responses, or error traces to external analytics servers. All evaluation data resides solely on the machine running the command.
- **Provider Confidentiality**: API traffic is directed strictly to your configured LLM endpoints (e.g. OpenAI, Anthropic, Bedrock, or local Ollama/vLLM instances).
- **Offline / Mock Mode**: Running with `--mock` executes 100% offline with zero outbound network requests.

### Data Retention & Cleanup Management
You maintain full control over disk storage and can sanitize historical data through multiple mechanisms:

1. **Automatic Pruning During Runs**:
   Pass `--db-retention-days <N>` to automatically prune runs older than $N$ days:
   ```bash
   promptdiff test v1.txt v2.txt --inputs cases.jsonl --db-retention-days 30
   ```
2. **Dedicated CLI Maintenance**:
   Prune historical records at any time:
   ```bash
   promptdiff db prune --days 14
   ```
3. **Complete Database Reset**:
   Wipe all historical records instantly:
   ```bash
   promptdiff db clear
   # or manually remove the database:
   rm -rf .promptdiff/telemetry.db
   ```
4. **Ephemeral Execution**:
   To avoid persisting data to disk entirely, route storage to in-memory SQLite:
   ```bash
   promptdiff test v1.txt v2.txt --inputs cases.jsonl --db-path ":memory:"
   ```
