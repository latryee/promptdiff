#!/usr/bin/env python3
"""GitHub Actions PR Commenter CLI Entrypoint for promptdiff."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from promptdiff.reporters.pr_bot import (
    generate_pr_markdown_comment,
    parse_pr_number_from_event,
    post_or_update_pr_comment,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Post PromptDiff evaluation report to GitHub Pull Request.")
    parser.add_argument("--report", "-r", default="report.json", help="Path to DiffReport JSON file")
    parser.add_argument(
        "--token", "-t", default=os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN"), help="GitHub API Token"
    )
    parser.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY"), help="Target repository (e.g. 'owner/repo')")
    parser.add_argument("--pr", "-p", type=int, default=None, help="Pull Request number")
    parser.add_argument(
        "--forecast",
        "-f",
        default=os.getenv("PROMPTDIFF_FORECAST"),
        help="Optional daily volume forecast (e.g. '1M', '500k')",
    )

    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.is_file():
        print(f"[!] Report file not found at {args.report}", file=sys.stderr)
        return 1

    try:
        report_data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[!] Failed to parse report JSON: {e}", file=sys.stderr)
        return 1

    token = args.token
    if not token:
        print("[!] Missing GITHUB_TOKEN. Set --token or GITHUB_TOKEN environment variable.", file=sys.stderr)
        return 1

    repo = args.repo
    if not repo:
        print("[!] Missing repository name. Set --repo or GITHUB_REPOSITORY environment variable.", file=sys.stderr)
        return 1

    pr_number = args.pr
    if not pr_number:
        event_path = os.getenv("GITHUB_EVENT_PATH")
        if event_path:
            pr_number = parse_pr_number_from_event(event_path)

    if not pr_number:
        print("[!] Could not determine Pull Request number. Set --pr argument.", file=sys.stderr)
        return 1

    comment_body = generate_pr_markdown_comment(report_data, forecast_vol=args.forecast)
    ok = post_or_update_pr_comment(token=token, repo=repo, pr_number=pr_number, body=comment_body)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
