"""Automated LLM Model Pricing Synchronization Script.

Fetches the latest open-source model pricing registry (e.g., from LiteLLM's public
catalog), computes price deltas against `promptdiff.pricing.MODEL_PRICING_TABLE`,
and generates structured diff reports or updates pricing tables.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from typing import Any

import httpx

from promptdiff.pricing import MODEL_PRICING_TABLE, ModelPrice

logger = logging.getLogger("sync_pricing")

LITELLM_PRICING_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"


@dataclass
class PriceDiff:
    """Represents a pricing discrepancy between local and remote registries."""

    model: str
    local_input: float
    remote_input: float
    local_output: float
    remote_output: float
    status: str  # "MODIFIED", "ADDED", "REMOVED"


def fetch_remote_pricing(url: str = LITELLM_PRICING_URL, timeout: float = 10.0) -> dict[str, Any]:
    """Fetch remote pricing JSON dictionary."""
    try:
        resp = httpx.get(url, timeout=timeout)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data
    except Exception as e:
        logger.warning(f"Could not fetch remote pricing from {url}: {e}")
        return {}


def compute_pricing_diff(
    local_table: dict[str, ModelPrice],
    remote_data: dict[str, Any],
) -> list[PriceDiff]:
    """Compare local ModelPrice entries against remote registry."""
    diffs: list[PriceDiff] = []

    for model_name, local_price in local_table.items():
        if model_name in ("mock", "ollama", "local"):
            continue

        remote_entry = remote_data.get(model_name)
        if not remote_entry and "/" not in model_name:
            # Try matching with provider prefix (e.g. openai/gpt-4o)
            for k, v in remote_data.items():
                if k.endswith(f"/{model_name}"):
                    remote_entry = v
                    break

        if not remote_entry or not isinstance(remote_entry, dict):
            continue

        # LiteLLM stores prices per single token (input_cost_per_token)
        rem_in_per_tok = float(remote_entry.get("input_cost_per_token", 0.0) or 0.0)
        rem_out_per_tok = float(remote_entry.get("output_cost_per_token", 0.0) or 0.0)

        rem_in_m = round(rem_in_per_tok * 1_000_000, 4)
        rem_out_m = round(rem_out_per_tok * 1_000_000, 4)

        if rem_in_m <= 0 and rem_out_m <= 0:
            continue

        in_diff = abs(local_price.input_per_million - rem_in_m) > 0.001
        out_diff = abs(local_price.output_per_million - rem_out_m) > 0.001

        if in_diff or out_diff:
            diffs.append(
                PriceDiff(
                    model=model_name,
                    local_input=local_price.input_per_million,
                    remote_input=rem_in_m,
                    local_output=local_price.output_per_million,
                    remote_output=rem_out_m,
                    status="MODIFIED",
                )
            )

    return diffs


def format_diff_table(diffs: list[PriceDiff]) -> str:
    """Format price differences as text table."""
    if not diffs:
        return "All model prices in promptdiff are in sync with remote registry."

    lines = [
        f"{'Model':<30} | {'Local (In/Out)':<20} | {'Remote (In/Out)':<20} | {'Status'}",
        "-" * 85,
    ]
    for d in diffs:
        loc = f"${d.local_input:.2f} / ${d.local_output:.2f}"
        rem = f"${d.remote_input:.2f} / ${d.remote_output:.2f}"
        lines.append(f"{d.model:<30} | {loc:<20} | {rem:<20} | {d.status}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync LLM model pricing with remote sources")
    parser.add_argument("--check", action="store_true", help="Exit with 1 if differences are detected")
    parser.add_argument("--json", action="store_true", help="Output diff as JSON")
    args = parser.parse_args()

    remote_data = fetch_remote_pricing()
    diffs = compute_pricing_diff(MODEL_PRICING_TABLE, remote_data)

    if args.json:
        print(json.dumps([d.__dict__ for d in diffs], indent=2))
    else:
        print(format_diff_table(diffs))

    if args.check and diffs:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
