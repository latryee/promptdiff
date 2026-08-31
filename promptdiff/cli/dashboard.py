"""Streamlit Web Dashboard for Interactive promptdiff Telemetry & Diff Visualization."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

try:
    import streamlit as st
    STREAMLIT_INSTALLED = True
except ImportError:
    STREAMLIT_INSTALLED = False


def launch_dashboard(port: int = 8501, host: str = "localhost", report_path: Optional[str] = None) -> None:
    """Launch Streamlit dashboard server."""
    if not STREAMLIT_INSTALLED:
        print("[!] Error: streamlit is not installed. Install with `pip install promptdiff[ui]`.", file=sys.stderr)
        sys.exit(1)

    app_path = Path(__file__).resolve()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.address",
        host,
    ]
    if report_path:
        cmd.extend(["--", report_path])

    print(f"[+] Launching promptdiff Streamlit Web Dashboard at http://{host}:{port} ...")
    subprocess.run(cmd)


def _get_mock_diff_report() -> dict[str, Any]:
    """Provide sample data for dashboard preview if no JSON report loaded."""
    return {
        "run_id": "run_demo_sample",
        "timestamp": "2026-08-29T11:00:00Z",
        "v1_name": "system_v1.txt",
        "v2_name": "system_v2.txt",
        "model_v1": "gpt-4o",
        "model_v2": "gpt-4o",
        "total_cases": 4,
        "evaluators": ["json_validity", "latency", "cost", "similarity", "llm_judge", "faithfulness", "security"],
        "verdict": {
            "passed": True,
            "status": "PASSED",
            "failed_assertions": [],
            "total_cost_v1": 0.000450,
            "total_cost_v2": 0.000315,
            "cost_delta_pct": -30.0,
            "avg_latency_v1": 240.5,
            "avg_latency_v2": 165.2,
            "latency_delta_pct": -31.3,
        },
        "aggregate_stats": {
            "passed_cases": 4,
            "total_cost_v1": 0.000450,
            "total_cost_v2": 0.000315,
            "cost_delta_pct": -30.0,
            "avg_latency_v1": 240.5,
            "avg_latency_v2": 165.2,
            "latency_delta_pct": -31.3,
        },
        "comparisons": [
            {
                "test_case": {
                    "id": "tc_01",
                    "description": "Password reset inquiry with security check",
                    "vars": {"query": "How do I reset my account password?", "context": "Password resets require 2FA authentication via Settings > Security."},
                },
                "v1_result": {
                    "output": "Dear customer, to reset your password, please contact support at support@example.com or visit settings.",
                    "latency_ms": 250.0,
                    "prompt_tokens": 30,
                    "completion_tokens": 25,
                    "total_tokens": 55,
                    "cost_usd": 0.00012,
                    "model": "gpt-4o",
                },
                "v2_result": {
                    "output": "To reset your password:\n1. Go to Settings > Security\n2. Click 'Reset Password'\n3. Verify via 2FA authentication.",
                    "latency_ms": 170.0,
                    "prompt_tokens": 25,
                    "completion_tokens": 20,
                    "total_tokens": 45,
                    "cost_usd": 0.00008,
                    "model": "gpt-4o",
                },
                "scores": {
                    "similarity": {"v1_score": 1.0, "v2_score": 0.88, "passed": True, "message": "88% Semantic Match"},
                    "llm_judge": {"v1_score": 4.0, "v2_score": 4.8, "passed": True, "message": "Judge: 4.8/5.0"},
                    "faithfulness": {"v1_score": 0.9, "v2_score": 1.0, "passed": True, "message": "100% Grounded in Context"},
                    "security": {"v1_score": 1.0, "v2_score": 1.0, "passed": True, "message": "Clean (0 PII, 0 Injections)"},
                },
            },
            {
                "test_case": {
                    "id": "tc_02",
                    "description": "Refund request within 30-day window",
                    "vars": {"query": "I want a refund for order #9821.", "context": "Refunds are eligible within 30 days of purchase upon submitting invoice."},
                },
                "v1_result": {
                    "output": "Refunds are processed manually by our finance department. Please send credit card details.",
                    "latency_ms": 230.0,
                    "prompt_tokens": 32,
                    "completion_tokens": 20,
                    "total_tokens": 52,
                    "cost_usd": 0.00011,
                    "model": "gpt-4o",
                },
                "v2_result": {
                    "output": "Orders within 30 days are eligible for a full refund. Please submit your order invoice in the portal.",
                    "latency_ms": 160.0,
                    "prompt_tokens": 28,
                    "completion_tokens": 18,
                    "total_tokens": 46,
                    "cost_usd": 0.00007,
                    "model": "gpt-4o",
                },
                "scores": {
                    "similarity": {"v1_score": 1.0, "v2_score": 0.82, "passed": True, "message": "82% Semantic Match"},
                    "llm_judge": {"v1_score": 3.5, "v2_score": 4.6, "passed": True, "message": "Judge: 4.6/5.0"},
                    "faithfulness": {"v1_score": 0.85, "v2_score": 1.0, "passed": True, "message": "100% Grounded in Context"},
                    "security": {"v1_score": 1.0, "v2_score": 1.0, "passed": True, "message": "Clean (0 PII, 0 Injections)"},
                },
            },
        ],
    }


def render_streamlit_app() -> None:
    """Streamlit Application Entrypoint."""
    st.set_page_config(
        page_title="PromptDiff v3.0 Studio",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .main { background-color: #0b0f19; }
        .metric-card { background-color: #151e2e; border: 1px solid #233149; border-radius: 10px; padding: 16px; margin-bottom: 12px; }
        .badge-pass { background-color: #064e3b; color: #6ee7b7; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 11px; }
        .badge-fail { background-color: #881337; color: #fda4af; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 11px; }
        .diff-box { font-family: monospace; font-size: 12px; line-height: 1.5; padding: 12px; border-radius: 8px; background: #0f172a; border: 1px solid #334155; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("⚡ PromptDiff v3.0 Enterprise Studio")
    st.caption("LLM Prompt Regression Testing, Multi-Model Arena, RAG Evaluators & Guardrails Dashboard")

    # Sidebar Data Loader
    st.sidebar.header("📂 Evaluation Data Source")
    uploaded_file = st.sidebar.file_uploader("Upload report.json or arena_report.json", type=["json"])

    report_data = None
    if uploaded_file is not None:
        try:
            report_data = json.load(uploaded_file)
            st.sidebar.success(f"Loaded: {uploaded_file.name}")
        except Exception as e:
            st.sidebar.error(f"Invalid JSON file: {e}")

    # Fallback to CLI argument or sample report
    if report_data is None:
        if len(sys.argv) > 1 and Path(sys.argv[-1]).is_file() and sys.argv[-1].endswith(".json"):
            try:
                report_data = json.loads(Path(sys.argv[-1]).read_text(encoding="utf-8"))
            except Exception:
                report_data = _get_mock_diff_report()
        else:
            report_data = _get_mock_diff_report()

    # Determine report type
    is_arena = "leaderboard" in report_data

    tabs = st.tabs([
        "📊 Executive Summary",
        "🔍 Side-by-Side Diff Inspector",
        "🏆 Multi-Model Arena",
        "🛡️ Security & Guardrails Audit",
        "🧠 Auto-Prompt Optimizer Studio",
    ])

    # TAB 1: Executive Summary
    with tabs[0]:
        v = report_data.get("verdict", {})
        status = v.get("status", "PASSED")
        passed = v.get("passed", True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Total Token Cost Delta",
                value=f"${v.get('total_cost_v2', 0.0):.6f}",
                delta=f"{v.get('cost_delta_pct', 0.0):+.1f}%",
                delta_color="inverse",
            )
        with col2:
            st.metric(
                label="Avg Latency Delta",
                value=f"{v.get('avg_latency_v2', 0.0):.1f} ms",
                delta=f"{v.get('latency_delta_pct', 0.0):+.1f}%",
                delta_color="inverse",
            )
        with col3:
            total_cases = report_data.get("total_cases", 0)
            passed_cases = report_data.get("aggregate_stats", {}).get("passed_cases", total_cases)
            st.metric(
                label="Test Cases Passing",
                value=f"{passed_cases} / {total_cases}",
                delta=f"{(passed_cases / total_cases * 100) if total_cases > 0 else 100:.0f}%",
            )
        with col4:
            st.metric(
                label="Quality Gate Verdict",
                value=status,
                delta="PASSED" if passed else "REGRESSION",
                delta_color="normal" if passed else "inverse",
            )

        st.markdown("---")
        st.subheader("📈 Performance Telemetry Comparison")
        comparisons = report_data.get("comparisons", [])
        if comparisons:
            case_ids = [c["test_case"]["id"] for c in comparisons]
            v1_latencies = [c.get("v1_result", {}).get("latency_ms", 0) for c in comparisons]
            v2_latencies = [c.get("v2_result", {}).get("latency_ms", 0) for c in comparisons]

            import pandas as pd
            df_chart = pd.DataFrame({
                "Test Case": case_ids,
                "v1 Baseline (ms)": v1_latencies,
                "v2 Candidate (ms)": v2_latencies,
            })
            st.bar_chart(df_chart.set_index("Test Case"))

    # TAB 2: Side-by-Side Diff Inspector
    with tabs[1]:
        st.subheader("🔍 Side-by-Side Response & Evaluator Breakdown")
        comparisons = report_data.get("comparisons", [])
        if not comparisons:
            st.info("No comparison results available in loaded report.")
        else:
            for idx, comp in enumerate(comparisons):
                tc = comp.get("test_case", {})
                v1_res = comp.get("v1_result", {})
                v2_res = comp.get("v2_result", {})
                scores = comp.get("scores", {})

                with st.expander(f"Case {tc.get('id')}: {tc.get('description', '')}", expanded=(idx == 0)):
                    if tc.get("vars"):
                        st.json(tc["vars"], expanded=False)

                    c_left, c_right = st.columns(2)
                    with c_left:
                        st.markdown(f"**Baseline (v1)** — `{v1_res.get('model', '')}` ({v1_res.get('latency_ms', 0):.0f}ms, ${v1_res.get('cost_usd', 0):.6f})")
                        st.markdown(f'<div class="diff-box">{v1_res.get("output", "")}</div>', unsafe_allow_html=True)
                    with c_right:
                        st.markdown(f"**Candidate (v2)** — `{v2_res.get('model', '')}` ({v2_res.get('latency_ms', 0):.0f}ms, ${v2_res.get('cost_usd', 0):.6f})")
                        st.markdown(f'<div class="diff-box">{v2_res.get("output", "")}</div>', unsafe_allow_html=True)

                    st.markdown("**Evaluator Scores:**")
                    score_cols = st.columns(max(1, len(scores)))
                    for sc_idx, (sc_name, sc_obj) in enumerate(scores.items()):
                        with score_cols[sc_idx]:
                            badge_class = "badge-pass" if sc_obj.get("passed", True) else "badge-fail"
                            st.markdown(
                                f'<div class="{badge_class}">{sc_name.replace("_", " ").title()}: {sc_obj.get("v2_score")}</div>'
                                f'<div style="font-size: 11px; color: #94a3b8;">{sc_obj.get("message", "")}</div>',
                                unsafe_allow_html=True,
                            )

    # TAB 3: Multi-Model Arena
    with tabs[2]:
        st.subheader("🏆 Multi-Model Arena Leaderboard (A/B/C/D)")
        if is_arena:
            leaderboard = report_data.get("leaderboard", [])
            st.dataframe(leaderboard, use_container_width=True)
        else:
            st.info("Load an `arena_report.json` or run `promptdiff arena` to populate the Multi-Model leaderboard.")
            st.code("promptdiff arena --prompts prompts/v1.txt,prompts/v2.txt --models gpt-4o,claude-3-5-sonnet,gemini-2.0-flash --mock")

    # TAB 4: Security & Guardrails
    with tabs[3]:
        st.subheader("🛡️ Enterprise Security & Guardrails Audit")
        st.markdown("Automated PII detection (Emails, Credit Cards, SSNs, API Secrets) & Prompt Injection Defense.")
        sec_findings = []
        for comp in report_data.get("comparisons", []):
            sec_score = comp.get("scores", {}).get("security", {})
            if sec_score:
                details = sec_score.get("details", {})
                sec_findings.append({
                    "Test Case": comp.get("test_case", {}).get("id"),
                    "Risk Level": details.get("risk_level", "CLEAN"),
                    "PII Leaks Found": len(details.get("v2_pii_findings", [])),
                    "Prompt Injection Breach": details.get("v2_injection_breach", False),
                    "Message": sec_score.get("message", "Clean"),
                })
        if sec_findings:
            st.dataframe(sec_findings, use_container_width=True)
        else:
            st.success("Zero security vulnerabilities or PII disclosures detected.")

    # TAB 5: Auto-Prompt Optimizer Studio
    with tabs[4]:
        st.subheader("🧠 Auto-Prompt Optimizer Studio (DSPy Style)")
        st.markdown("Reflectively optimize failing prompts using Meta-LLM reasoning feedback.")
        st.code("promptdiff optimize prompts/system_v1.txt --inputs testcases.jsonl --output prompts/system_v3_optimized.txt")


if __name__ == "__main__" and STREAMLIT_INSTALLED:
    render_streamlit_app()
