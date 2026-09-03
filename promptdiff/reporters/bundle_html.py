"""Zero-Dependency Interactive Standalone HTML Report Bundle for promptdiff."""

from __future__ import annotations

from pathlib import Path

from promptdiff.core.models import DiffReport
from promptdiff.core.statistics import analyze_significance
from promptdiff.pricing import calculate_forecast

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>⚡ PromptDiff Interactive Report - {v1_name} vs {v2_name}</title>
  <style>
    :root {{
      --bg: #0b0f19;
      --card-bg: #0f172a;
      --border: #1e293b;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --success: #10b981;
      --danger: #ef4444;
      --warning: #f59e0b;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 2rem;
    }}
    .container {{ max-width: 1300px; margin: 0 auto; }}
    header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid var(--border); }}
    .title {{ font-size: 1.8rem; font-weight: 800; color: var(--accent); }}
    .badge {{ display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: bold; font-size: 0.85rem; }}
    .badge-pass {{ background: rgba(16, 185, 129, 0.2); color: var(--success); border: 1px solid var(--success); }}
    .badge-fail {{ background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid var(--danger); }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1.25rem; margin-bottom: 2rem; }}
    .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 0.75rem; padding: 1.5rem; }}
    .card-title {{ font-size: 0.85rem; text-transform: uppercase; color: var(--text-muted); font-weight: 600; margin-bottom: 0.5rem; }}
    .card-val {{ font-size: 1.6rem; font-weight: 700; }}
    .diff-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }}
    .diff-box {{ background: #020617; border: 1px solid var(--border); border-radius: 0.5rem; padding: 1rem; font-family: monospace; font-size: 0.9rem; white-space: pre-wrap; }}
    .search-bar {{ width: 100%; padding: 0.75rem 1rem; border-radius: 0.5rem; border: 1px solid var(--border); background: #020617; color: var(--text); font-size: 1rem; margin-bottom: 1.5rem; }}
    .case-card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 0.5rem; padding: 1.25rem; margin-bottom: 1rem; }}
    .tag {{ background: #1e293b; color: #94a3b8; padding: 0.2rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <div class="title">⚡ PromptDiff Interactive Regression Report</div>
        <p style="color: var(--text-muted); margin-top: 0.25rem;">
          <b>{v1_name}</b> ➔ <b>{v2_name}</b> &nbsp;|&nbsp; Model: <code>{model}</code>
        </p>
      </div>
      <div>
        <span class="badge {status_class}">{status_text}</span>
      </div>
    </header>

    <div class="kpi-grid">
      <div class="card">
        <div class="card-title">Token Cost Impact</div>
        <div class="card-val" style="color: {cost_color};">{cost_delta_pct}</div>
        <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.25rem;">${cost_v1:.6f} ➔ ${cost_v2:.6f}</div>
      </div>
      <div class="card">
        <div class="card-title">Latency Impact</div>
        <div class="card-val" style="color: {lat_color};">{lat_delta_pct}</div>
        <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.25rem;">{lat_v1:.1f}ms ➔ {lat_v2:.1f}ms</div>
      </div>
      <div class="card">
        <div class="card-title">Test Cases Passed</div>
        <div class="card-val" style="color: var(--success);">{passed_cases} / {total_cases}</div>
        <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.25rem;">{pass_rate}% Success Rate</div>
      </div>
      <div class="card">
        <div class="card-title">Monthly Volume Projection (1M reqs/day)</div>
        <div class="card-val" style="color: var(--accent); font-size: 1.25rem;">{monthly_savings}</div>
        <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.25rem;">Scale Cost Analysis</div>
      </div>
    </div>

    <div class="card" style="margin-bottom: 2rem;">
      <div class="card-title">🔬 Statistical Significance Analysis (Bootstrap 95% Confidence Intervals)</div>
      <p style="margin-top: 0.5rem; font-size: 1rem; color: var(--text);">{significance_summary}</p>
    </div>

    <input type="text" id="searchInput" class="search-bar" placeholder="🔍 Search test cases by ID, description, or keyword..." onkeyup="filterCases()">

    <div id="casesList">
      {case_cards}
    </div>
  </div>

  <script>
    function filterCases() {{
      const query = document.getElementById('searchInput').value.toLowerCase();
      const cards = document.getElementsByClassName('case-card');
      for (let card of cards) {{
        const text = card.innerText.toLowerCase();
        card.style.display = text.includes(query) ? 'block' : 'none';
      }}
    }}
  </script>
</body>
</html>
"""


def generate_interactive_bundle_html(
    report: DiffReport,
    output_path: str = "promptdiff-bundle.html",
    forecast_volume: str | int = 1_000_000,
) -> str:
    """Generate self-contained, single-file interactive HTML report."""
    v = report.verdict
    total = report.total_cases
    passed_cases = report.aggregate_stats.get("passed_cases", total)
    pass_rate = round((passed_cases / total * 100.0) if total > 0 else 100.0, 1)

    fc = calculate_forecast(v.total_cost_v1, v.total_cost_v2, total, forecast_volume)
    savings_text = (
        f"+${fc.monthly_savings_usd:,.2f}/mo Savings"
        if fc.monthly_savings_usd > 0
        else f"${fc.monthly_delta_cost:+,.2f}/mo"
    )

    # Statistical significance on latency
    v1_lats = [float(comp.v1_result.latency_ms) for comp in report.comparisons]
    v2_lats = [float(comp.v2_result.latency_ms) for comp in report.comparisons]
    sig = analyze_significance("latency_ms", v1_lats, v2_lats)
    sig_text = sig.verdict_text if sig else "Sample size insufficient for statistical bootstrap."

    # Build case cards
    cards = []
    for comp in report.comparisons:
        tc = comp.test_case
        all_passed = all(s.passed for s in comp.scores.values())
        badge = (
            '<span class="badge badge-pass">PASS</span>' if all_passed else '<span class="badge badge-fail">FAIL</span>'
        )

        scores_badges = " ".join(f'<span class="tag">{k}: {s.v2_score}</span>' for k, s in comp.scores.items())

        card_html = f"""
        <div class="case-card">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <div>
              <span style="font-weight: 700; font-size: 1.05rem;">{tc.id}</span>
              <span style="color: var(--text-muted); margin-left: 0.75rem;">{tc.description}</span>
            </div>
            <div>{badge}</div>
          </div>
          <div style="margin-bottom: 0.75rem;">{scores_badges}</div>
          <div class="diff-container">
            <div class="diff-box"><b style="color: var(--accent);">Baseline ({report.v1_name}):</b>\n\n{comp.v1_result.output}</div>
            <div class="diff-box"><b style="color: #c084fc;">Candidate ({report.v2_name}):</b>\n\n{comp.v2_result.output}</div>
          </div>
        </div>
        """
        cards.append(card_html)

    html_content = HTML_TEMPLATE.format(
        v1_name=report.v1_name,
        v2_name=report.v2_name,
        model=report.model_v2,
        status_text="ALL GATES PASSED" if v.passed else "REGRESSION DETECTED",
        status_class="badge-pass" if v.passed else "badge-fail",
        cost_delta_pct=f"{v.cost_delta_pct:+.1f}%",
        cost_color="var(--success)" if v.cost_delta_pct <= 0 else "var(--danger)",
        cost_v1=v.total_cost_v1,
        cost_v2=v.total_cost_v2,
        lat_delta_pct=f"{v.latency_delta_pct:+.1f}%",
        lat_color="var(--success)" if v.latency_delta_pct <= 0 else "var(--danger)",
        lat_v1=v.avg_latency_v1,
        lat_v2=v.avg_latency_v2,
        passed_cases=passed_cases,
        total_cases=total,
        pass_rate=pass_rate,
        monthly_savings=savings_text,
        significance_summary=sig_text,
        case_cards="\n".join(cards),
    )

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html_content, encoding="utf-8")
    return str(target.resolve())
