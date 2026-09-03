"""Coverage tests for production canary rollout and streaming profiler."""

from __future__ import annotations

from pathlib import Path

import pytest

from promptdiff.core.models import DiffReport, PromptVersion, RegressionVerdict
from promptdiff.production.canary import CanaryConfigGenerator
from promptdiff.production.profiler import StreamingProfiler, StreamingProfileResult


def _make_dummy_report(passed: bool, cost_delta_pct: float = 0.0, latency_delta_pct: float = 0.0) -> DiffReport:
    verdict = RegressionVerdict(
        passed=passed,
        cost_delta_pct=cost_delta_pct,
        latency_delta_pct=latency_delta_pct,
    )
    return DiffReport(
        run_id="test_run",
        v1_name="v1",
        v2_name="v2",
        model_v1="mock-gpt-4o",
        model_v2="mock-gpt-4o",
        verdict=verdict,
        comparisons=[],
        evaluators=[],
        total_cases=1,
    )


def test_canary_rollout_hold_on_failure() -> None:
    rep = _make_dummy_report(passed=False)
    gen = CanaryConfigGenerator(rep, flag_name="test_flag")
    cfg = gen.generate()

    assert cfg.v1_weight_pct == 100
    assert cfg.v2_weight_pct == 0
    assert "HOLD" in cfg.recommendation
    assert cfg.launchdarkly_json["key"] == "test_flag"
    assert cfg.statsig_json["rules"][0]["passPercentage"] == 0
    assert cfg.openfeature_json["flags"]["test_flag"]["defaultVariant"] == "v1"


def test_canary_rollout_accelerated() -> None:
    rep = _make_dummy_report(passed=True, cost_delta_pct=-15.0, latency_delta_pct=-5.0)
    gen = CanaryConfigGenerator(rep)
    cfg = gen.generate()

    assert cfg.v1_weight_pct == 50
    assert cfg.v2_weight_pct == 50
    assert "ACCELERATED" in cfg.recommendation
    assert cfg.openfeature_json["flags"]["prompt_system_v2_rollout"]["defaultVariant"] == "v2"


def test_canary_rollout_standard_safe() -> None:
    rep = _make_dummy_report(passed=True, cost_delta_pct=2.0, latency_delta_pct=1.0)
    gen = CanaryConfigGenerator(rep)
    cfg = gen.generate()

    assert cfg.v1_weight_pct == 90
    assert cfg.v2_weight_pct == 10
    assert "SAFE" in cfg.recommendation


def test_canary_save_to_file(tmp_path: Path) -> None:
    rep = _make_dummy_report(passed=True)
    gen = CanaryConfigGenerator(rep)
    cfg = gen.generate()

    target = tmp_path / "canary.json"
    path = gen.save_to_file(cfg, str(target))
    assert Path(path).exists()
    content = Path(path).read_text(encoding="utf-8")
    assert "launchdarkly" in content


@pytest.mark.asyncio
async def test_streaming_profiler_mock_mode() -> None:
    pv = PromptVersion(name="v1", template="Help: {{query}}")
    profiler = StreamingProfiler(prompt_version=pv, force_mock=True)
    result = await profiler.profile_stream("test query")

    assert isinstance(result, StreamingProfileResult)
    assert result.time_to_first_token_ms > 0.0
    assert result.tokens_per_second > 0.0
    assert "Mock streaming" in result.full_output


@pytest.mark.asyncio
async def test_streaming_profiler_with_provider() -> None:
    pv = PromptVersion(name="v1", template="Answer: {{query}}")
    profiler = StreamingProfiler(prompt_version=pv, force_mock=False, model_name="mock-gpt-4o")
    result = await profiler.profile_stream("hello")

    assert isinstance(result, StreamingProfileResult)
    assert result.total_latency_ms > 0.0
    assert result.total_tokens_received > 0
