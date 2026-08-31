"""A/B/n Canary Rollout & Feature Flag Config Generator for promptdiff (promptdiff canary)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from promptdiff.core.models import DiffReport


@dataclass
class CanaryRolloutConfig:
    """Multi-platform feature flag & canary rollout configuration."""

    flag_key: str
    v1_weight_pct: int
    v2_weight_pct: int
    recommendation: str
    launchdarkly_json: dict[str, Any] = field(default_factory=dict)
    statsig_json: dict[str, Any] = field(default_factory=dict)
    openfeature_json: dict[str, Any] = field(default_factory=dict)


class CanaryConfigGenerator:
    """Generates production A/B/n feature flag configs based on regression evaluation verdict."""

    def __init__(self, report: DiffReport, flag_name: str = "prompt_system_v2_rollout"):
        self.report = report
        self.flag_name = flag_name

    def generate(self) -> CanaryRolloutConfig:
        v = self.report.verdict

        # Calculate recommended rollout percentage
        if not v.passed:
            # Quality gate failure -> 0% rollout (Hold)
            v1_w = 100
            v2_w = 0
            rec = "HOLD (0% Rollout): Regressions detected. Candidate does not meet safety/cost thresholds."
        elif v.cost_delta_pct <= -10.0 and v.latency_delta_pct <= 0:
            # Major improvement -> 50% or 100% rollout
            v1_w = 50
            v2_w = 50
            rec = "ACCELERATED CANARY (50% Rollout): Significant cost and latency reduction with zero quality regressions."
        else:
            # Standard safe canary rollout -> 10%
            v1_w = 90
            v2_w = 10
            rec = "SAFE CANARY (10% Rollout): Quality assertions passed. Start with 10% canary traffic."

        # LaunchDarkly format
        ld_config = {
            "key": self.flag_name,
            "name": "Prompt Version Rollout",
            "variations": [
                {"value": self.report.v1_name, "name": "Baseline (v1)"},
                {"value": self.report.v2_name, "name": "Candidate (v2)"},
            ],
            "fallthrough": {
                "rollout": {
                    "variations": [
                        {"variation": 0, "weight": v1_w * 1000},
                        {"variation": 1, "weight": v2_w * 1000},
                    ]
                }
            },
        }

        # Statsig Dynamic Config format
        statsig_config = {
            "name": self.flag_name,
            "type": "feature_gate",
            "defaultValue": False,
            "rules": [
                {
                    "name": "Canary Split",
                    "passPercentage": v2_w,
                    "return": {"version": self.report.v2_name},
                }
            ],
        }

        # OpenFeature Flagd format
        openfeature_config = {
            "flags": {
                self.flag_name: {
                    "state": "ENABLED",
                    "variants": {
                        "v1": self.report.v1_name,
                        "v2": self.report.v2_name,
                    },
                    "defaultVariant": "v1" if v2_w == 0 else "v2",
                }
            }
        }

        return CanaryRolloutConfig(
            flag_key=self.flag_name,
            v1_weight_pct=v1_w,
            v2_weight_pct=v2_w,
            recommendation=rec,
            launchdarkly_json=ld_config,
            statsig_json=statsig_config,
            openfeature_json=openfeature_config,
        )

    def save_to_file(self, config: CanaryRolloutConfig, output_path: str = "canary_config.json") -> str:
        """Export canary configs to JSON file."""
        target = Path(output_path)
        payload = {
            "flag_key": config.flag_key,
            "allocation": {
                "v1_baseline_pct": config.v1_weight_pct,
                "v2_candidate_pct": config.v2_weight_pct,
            },
            "recommendation": config.recommendation,
            "providers": {
                "launchdarkly": config.launchdarkly_json,
                "statsig": config.statsig_json,
                "openfeature": config.openfeature_json,
            },
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(target.resolve())
