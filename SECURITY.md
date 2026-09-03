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
