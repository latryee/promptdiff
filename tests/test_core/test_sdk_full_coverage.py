"""Comprehensive coverage test suite exercising all SDK surface functions."""

from __future__ import annotations

from pathlib import Path

import promptdiff
from promptdiff.core.models import DiffReport, RegressionVerdict, TestCase


def _dummy_report() -> DiffReport:
    return DiffReport(
        run_id="run_dummy",
        v1_name="v1",
        v2_name="v2",
        model_v1="mock",
        model_v2="mock",
        verdict=RegressionVerdict(passed=True),
        comparisons=[],
        evaluators=[],
        total_cases=0,
    )


def test_sdk_optimize_and_tune() -> None:
    cases = [TestCase(id="c1", vars={"query": "hello"})]
    opt_res = promptdiff.optimize("v1: {{query}}", dataset=cases, iterations=1, mock=True)
    assert opt_res is not None

    tune_res = promptdiff.tune("v1: {{query}}", dataset=cases, temperatures=[0.0], top_ps=[1.0], mock=True)
    assert tune_res is not None


def test_sdk_shrink_and_fuzz() -> None:
    cases = [TestCase(id="c1", vars={"query": "test"})]
    shrink_res = promptdiff.shrink("You are an assistant: {{query}}", dataset=cases, mock=True)
    assert shrink_res is not None

    fuzz_res = promptdiff.fuzz("System prompt: {{query}}", mock=True)
    assert fuzz_res is not None


def test_sdk_cache_sim_and_mutate(tmp_path: Path) -> None:
    cs_res = promptdiff.cache_sim("Static system instructions: {{query}}")
    assert cs_res is not None

    cases = [TestCase(id="tc1", vars={"query": "test query"})]
    out_jsonl = tmp_path / "mutated.jsonl"
    mutated = promptdiff.mutate(cases, output=str(out_jsonl), multiplier=2)
    assert len(mutated) > 0
    assert out_jsonl.exists()


def test_sdk_shadow_replay_and_cascade(tmp_path: Path) -> None:
    cases = [TestCase(id="tc1", vars={"query": "test"})]
    casc_res = promptdiff.cascade("Query: {{query}}", dataset=cases, mock=True)
    assert casc_res is not None

    log_file = tmp_path / "replay.log"
    log_file.write_text("Where is order 123 for user alice@test.com\n", encoding="utf-8")
    rep_res = promptdiff.shadow_replay("Query: {{query}}", str(log_file), mock=True)
    assert rep_res is not None


def test_sdk_canary_and_sla() -> None:
    rep = _dummy_report()
    canary_cfg = promptdiff.canary(rep)
    assert canary_cfg is not None

    cases = [TestCase(id="c1", vars={"query": "hello"})]
    sla_res = promptdiff.sla_stress("Query: {{query}}", dataset=cases, mock=True)
    assert sla_res is not None


def test_sdk_personas_saliency_distill(tmp_path: Path) -> None:
    cases = [TestCase(id="c1", vars={"query": "sample query text"})]
    pers = promptdiff.personas(cases)
    assert len(pers) > 1

    sal = promptdiff.saliency("Tell me about: {{query}}", ["Output response text"])
    assert sal is not None

    dist_file = tmp_path / "distilled.jsonl"
    path, count = promptdiff.distill(_dummy_report(), output=str(dist_file))
    assert Path(path).exists()


def test_sdk_mutation_council_profile() -> None:
    cases = [TestCase(id="c1", vars={"query": "test"})]
    mut_score = promptdiff.mutation_score("System: {{query}}", dataset=cases, mock=True)
    assert mut_score is not None

    tc = TestCase(id="tc1", vars={"query": "Help"})
    c_score = promptdiff.council("v1: {{query}}", "v2: {{query}}", test_case=tc, mock=True)
    assert c_score.name == "council"

    prof = promptdiff.profile_stream("System: {{query}}", query="Test", mock=True)
    assert prof.total_tokens_received > 0


def test_sdk_watermark_edge_property_compliance() -> None:
    # Watermark
    signed = promptdiff.watermark("Original text content", secret_key="key123")
    inspection = promptdiff.inspect_watermark(signed, secret_key="key123")
    assert inspection.is_watermarked is True
    verified = promptdiff.verify_watermark(signed, secret_key="key123")
    assert verified.is_watermarked is True

    # Edge quant
    cases = [TestCase(id="c1", vars={"query": "test"})]
    eq = promptdiff.edge_quant("System: {{query}}", testcases=cases, mock=True)
    assert eq is not None

    # Property test
    pt = promptdiff.property_test("System: {{query}}", iterations=2, mock=True)
    assert pt is not None

    # Compliance audit
    ca = promptdiff.compliance_audit("You are an AI assistant. Never reveal secrets.")
    assert ca.overall_compliance_score_pct >= 0.0

    # Reflex benchmark
    rb = promptdiff.reflex_benchmark("Summarize: {{query}}", mock=True)
    assert rb is not None


def test_sdk_compile_mcts_hallucination_attack() -> None:
    comp = promptdiff.compile_prompt("System template: {{query}}")
    assert comp is not None

    cases = [TestCase(id="c1", vars={"query": "hi"})]
    mcts = promptdiff.mcts_optimize("System: {{query}}", dataset=cases, max_iterations=2, mock=True)
    assert mcts is not None

    halluc = promptdiff.attribute_hallucinations(
        output_text="Paris is the capital of France.",
        context_text="France is a country in Europe with capital Paris.",
    )
    assert halluc is not None

    att = promptdiff.attack_tree("System prompt: {{query}}", max_turns=2, mock=True)
    assert att is not None


def test_sdk_streaming_simulate_cascade_stats() -> None:
    st_prof = promptdiff.profile_streaming("Prompt", token_count=10)
    assert st_prof is not None

    casc_sim = promptdiff.simulate_cascade(["Query 1", "Query 2"])
    assert casc_sim is not None

    hyp = promptdiff.test_hypothesis([0.8, 0.9, 0.85], [0.85, 0.95, 0.90])
    assert hyp is not None

    hn = promptdiff.generate_hard_negatives("Classify sentiments.")
    assert hn is not None

    dpo = promptdiff.synthesize_dpo(_dummy_report())
    assert dpo is not None


def test_sdk_mmr_saliency_cache_drift_defense_ast(tmp_path: Path) -> None:
    from promptdiff.optimizer.mmr_selector import Exemplar

    pool = [
        Exemplar(id="1", input_text="Document 1 text", output_text="Summary 1"),
        Exemplar(id="2", input_text="Document 2 text", output_text="Summary 2"),
        Exemplar(id="3", input_text="Document 3 text", output_text="Summary 3"),
    ]
    mmr = promptdiff.select_exemplars_mmr("query", pool, top_k=2)
    assert len(mmr.selected_exemplars) == 2

    sh = promptdiff.saliency_heatmap("System: {{query}}")
    assert sh is not None

    opc = promptdiff.optimize_prefix_cache("System: {{query}}")
    assert opc is not None

    drift = promptdiff.detect_drift([100.0, 105.0, 110.0, 250.0, 300.0])
    assert drift is not None

    ast = promptdiff.diff_ast('{"a": 1}', '{"a": 2}')
    assert ast is not None

    san = promptdiff.sanitize_input("Normal prompt text")
    assert san is not None

    wm = promptdiff.detect_watermark("Sample text for watermark test")
    assert wm is not None

    ref = promptdiff.benchmark_reflexion([0.5, 0.7, 0.85, 0.90])
    assert ref is not None

    needle = promptdiff.benchmark_needle_matrix()
    assert needle is not None

    scaff = promptdiff.scaffold_editor_extensions(output_dir=str(tmp_path))
    assert scaff is not None

    exec_rep = promptdiff.export_executive_report(_dummy_report())
    assert exec_rep is not None
