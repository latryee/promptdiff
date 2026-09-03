"""GitHub PR Commenter Bot engine for promptdiff."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import httpx

STICKY_HEADER_TAG = "<!-- promptdiff-ci-comment -->"


def parse_pr_number_from_event(event_path: str) -> Optional[int]:
    """Extract Pull Request number from GITHUB_EVENT_PATH JSON payload."""
    try:
        path = Path(event_path)
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if "pull_request" in data and "number" in data["pull_request"]:
                return int(data["pull_request"]["number"])
            elif "issue" in data and "number" in data["issue"]:
                return int(data["issue"]["number"])
            elif "number" in data:
                return int(data["number"])
    except Exception as e:
        print(f"[!] Could not parse PR number from event payload: {e}", file=sys.stderr)
    return None


def generate_pr_markdown_comment(report_data: dict[str, Any], forecast_vol: Optional[str] = None) -> str:
    """Construct full GitHub Flavored Markdown comment body from DiffReport dict."""
    v = report_data.get("verdict", {})
    passed = v.get("passed", True)
    total_cases = report_data.get("total_cases", 0)
    passed_cases = report_data.get("aggregate_stats", {}).get("passed_cases", total_cases)

    # Status Banner
    if passed:
        banner = "### ✅ **PromptDiff CI: All Quality Gates Passed**"
        badge = "https://img.shields.io/badge/Quality_Gate-PASSED-brightgreen.svg"
    else:
        banner = "### ❌ **PromptDiff CI: Regression Detected (Merge Blocked)**"
        badge = "https://img.shields.io/badge/Quality_Gate-REGRESSION_DETECTED-red.svg"

    cost_v1 = v.get("total_cost_v1", 0.0)
    cost_v2 = v.get("total_cost_v2", 0.0)
    cost_delta_pct = v.get("cost_delta_pct", 0.0)

    lat_v1 = v.get("avg_latency_v1", 0.0)
    lat_v2 = v.get("avg_latency_v2", 0.0)
    lat_delta_pct = v.get("latency_delta_pct", 0.0)

    cost_icon = "🟢" if cost_delta_pct <= 0 else "🔴"
    lat_icon = "🟢" if lat_delta_pct <= 0 else "🔴"

    # Optional Production Volume Forecasting
    forecast_row = ""
    if forecast_vol:
        from promptdiff.pricing import calculate_forecast

        fc = calculate_forecast(cost_v1, cost_v2, total_cases, forecast_vol)
        savings_text = (
            f"**+${fc.monthly_savings_usd:,.2f}/mo** ({fc.cost_delta_pct:+.1f}%)"
            if fc.monthly_savings_usd > 0
            else f"${fc.monthly_delta_cost:+,.2f}/mo"
        )
        forecast_row = f"| **Projected Monthly Impact ({fc.daily_volume:,} reqs/day)** | {savings_text} |\n"

    # Failed Assertions block
    failures_block = ""
    failed_assertions = v.get("failed_assertions", [])
    if failed_assertions:
        items = "\n".join(f"- ⚠️ `{f}`" for f in failed_assertions)
        failures_block = f"\n#### 🚨 Failed Assertions & Regressions:\n{items}\n"

    # Evaluator metrics summary table
    evaluators_rows = []
    comparisons = report_data.get("comparisons", [])
    eval_names = report_data.get("evaluators", [])

    for ev_name in eval_names:
        scores_v1 = []
        scores_v2 = []
        for comp in comparisons:
            s_obj = comp.get("scores", {}).get(ev_name)
            if s_obj and "v2_score" in s_obj and isinstance(s_obj["v2_score"], (int, float)):
                scores_v2.append(float(s_obj["v2_score"]))
            if s_obj and "v1_score" in s_obj and isinstance(s_obj["v1_score"], (int, float)):
                scores_v1.append(float(s_obj["v1_score"]))

        if scores_v2:
            avg_v1 = sum(scores_v1) / len(scores_v1) if scores_v1 else 0.0
            avg_v2 = sum(scores_v2) / len(scores_v2)
            delta = avg_v2 - avg_v1
            status_tag = (
                "✅ Pass"
                if all(c.get("scores", {}).get(ev_name, {}).get("passed", True) for c in comparisons)
                else "❌ Regressed"
            )
            evaluators_rows.append(f"| `{ev_name}` | `{avg_v1:.3f}` | `{avg_v2:.3f}` | `{delta:+.3f}` | {status_tag} |")

    eval_table = "\n".join(evaluators_rows)

    # Collapsible Diff Preview
    diff_cards = []
    for comp in comparisons[:5]:
        tc = comp.get("test_case", {})
        v1_out = comp.get("v1_result", {}).get("output", "")[:200]
        v2_out = comp.get("v2_result", {}).get("output", "")[:200]
        diff_cards.append(
            f"**Case `{tc.get('id')}`**: {tc.get('description', '')}\n\n"
            f"- **Baseline (v1)**: `{v1_out}...`\n"
            f"- **Candidate (v2)**: `{v2_out}...`\n"
        )
    diff_preview = "\n---\n".join(diff_cards)

    comment = f"""{STICKY_HEADER_TAG}
{banner}

<p align="left">
  <img src="{badge}" alt="Quality Gate Status" />
  &nbsp;
  <code>{report_data.get("v1_name", "v1")}</code> ➔ <code>{report_data.get("v2_name", "v2")}</code>
  &nbsp;|&nbsp;
  Model: <code>{report_data.get("model_v2", "gpt-4o")}</code>
</p>

### 📊 Regression KPI Summary
| Metric | Baseline (v1) | Candidate (v2) | Delta / Variance |
| :--- | :--- | :--- | :--- |
| **Total Token Cost** | `${cost_v1:.6f}` | `${cost_v2:.6f}` | {cost_icon} **`{cost_delta_pct:+.1f}%`** |
| **Avg Latency** | `{lat_v1:.1f} ms` | `{lat_v2:.1f} ms` | {lat_icon} **`{lat_delta_pct:+.1f}%`** |
| **Test Scenarios Passing** | `{total_cases} total` | `{passed_cases} passed` | **`{(passed_cases / total_cases * 100) if total_cases > 0 else 100:.0f}% Pass Rate`** |
{forecast_row}
{failures_block}

### 🔍 Evaluator Score Breakdown
| Evaluator | Baseline Score | Candidate Score | Delta | Verdict |
| :--- | :--- | :--- | :--- | :--- |
{eval_table}

<details>
<summary><b>🔎 Click to preview Side-by-Side Test Case Diffs</b></summary>

{diff_preview}

</details>

---
*Automated Prompt Regression Quality Gate generated by [promptdiff](https://github.com/latryee/promptdiff)*
"""
    return comment.strip()


def post_or_update_pr_comment(
    token: str,
    repo: str,
    pr_number: int,
    body: str,
) -> bool:
    """Post new PR comment or update existing sticky comment."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    base_url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"

    with httpx.Client(headers=headers, timeout=15.0) as client:
        existing_comment_id = None
        try:
            resp = client.get(base_url)
            if resp.status_code == 200:
                comments = resp.json()
                for c in comments:
                    if STICKY_HEADER_TAG in c.get("body", ""):
                        existing_comment_id = c.get("id")
                        break
        except Exception as e:
            print(f"[!] Warning: Could not search existing comments: {e}", file=sys.stderr)

        if existing_comment_id:
            update_url = f"https://api.github.com/repos/{repo}/issues/comments/{existing_comment_id}"
            patch_resp = client.patch(update_url, json={"body": body})
            if patch_resp.status_code == 200:
                print(f"[+] Successfully updated sticky PR comment #{existing_comment_id} on {repo}#{pr_number}")
                return True
            else:
                print(f"[!] Failed to update comment: {patch_resp.text}", file=sys.stderr)
        else:
            post_resp = client.post(base_url, json={"body": body})
            if post_resp.status_code in (200, 201):
                print(f"[+] Successfully created new PR comment on {repo}#{pr_number}")
                return True
            else:
                print(f"[!] Failed to create comment: {post_resp.text}", file=sys.stderr)

    return False
