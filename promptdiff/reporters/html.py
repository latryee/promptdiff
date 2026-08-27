"""Interactive Standalone HTML Report Generator for promptdiff."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import List
from promptdiff.core.models import ComparisonResult, DiffChunk, DiffReport


def _chunk_to_html(chunk: DiffChunk, side: str) -> str:
    """Convert DiffChunk to colored HTML span."""
    if chunk.kind == "equal":
        txt = chunk.v1_text if side == "v1" else chunk.v2_text
        return html.escape(txt)
    elif chunk.kind == "delete":
        if side == "v1":
            return f'<span class="bg-red-950 text-red-300 px-0.5 rounded font-semibold border border-red-800">{html.escape(chunk.v1_text)}</span>'
        return ""
    elif chunk.kind == "insert":
        if side == "v2":
            return f'<span class="bg-emerald-950 text-emerald-300 px-0.5 rounded font-semibold border border-emerald-800">{html.escape(chunk.v2_text)}</span>'
        return ""
    elif chunk.kind == "replace":
        if side == "v1":
            return f'<span class="bg-rose-950 text-rose-300 px-0.5 rounded font-semibold border border-rose-800">{html.escape(chunk.v1_text)}</span>'
        return f'<span class="bg-teal-950 text-teal-300 px-0.5 rounded font-semibold border border-teal-800">{html.escape(chunk.v2_text)}</span>'
    return ""


def generate_html_report(report: DiffReport, output_path: str) -> str:
    """Generate self-contained interactive dark-mode HTML report."""
    v = report.verdict
    status_bg = "bg-emerald-500/10 text-emerald-400 border-emerald-500/30" if v.passed else "bg-rose-500/10 text-rose-400 border-rose-500/30"
    status_icon = "✓" if v.passed else "⚠"
    status_text = "ALL ASSERTIONS PASSED" if v.passed else "REGRESSION DETECTED"

    cards_html = []
    for comp in report.comparisons:
        tc = comp.test_case
        v1_html = "".join(_chunk_to_html(c, "v1") for c in comp.text_diff)
        v2_html = "".join(_chunk_to_html(c, "v2") for c in comp.text_diff)

        # Scores badges
        score_badges = []
        all_passed = True
        for m_name, s in comp.scores.items():
            if not s.passed:
                all_passed = False
            badge_color = "bg-emerald-900/60 text-emerald-300 border-emerald-700" if s.passed else "bg-rose-900/60 text-rose-300 border-rose-700"
            score_badges.append(
                f'<div class="flex items-center justify-between text-xs px-2.5 py-1.5 rounded border {badge_color}">'
                f'<span class="font-medium">{html.escape(m_name.replace("_", " ").title())}</span>'
                f'<span class="font-mono">{html.escape(s.message)}</span>'
                f'</div>'
            )

        card_filter_class = "case-passed" if all_passed else "case-failed"

        vars_str = json.dumps(tc.vars, indent=2) if tc.vars else "{}"

        card = f"""
        <div class="test-card {card_filter_class} bg-slate-900 border border-slate-800 rounded-xl p-5 mb-6 shadow-xl">
            <div class="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
                <div>
                    <h3 class="text-base font-semibold text-slate-100 flex items-center gap-2">
                        <span class="px-2 py-0.5 bg-indigo-500/20 text-indigo-300 text-xs rounded border border-indigo-500/30 font-mono">{html.escape(tc.id)}</span>
                        {html.escape(tc.description or 'Regression test scenario')}
                    </h3>
                </div>
                <div class="text-xs text-slate-400 font-mono">
                    v1: {comp.v1_result.latency_ms:.0f}ms (${comp.v1_result.cost_usd:.5f}) &rarr; 
                    v2: {comp.v2_result.latency_ms:.0f}ms (${comp.v2_result.cost_usd:.5f})
                </div>
            </div>

            <!-- Side by Side Diff Panes -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4 font-mono text-xs">
                <!-- Left: V1 -->
                <div class="bg-slate-950/80 rounded-lg p-4 border border-slate-800/80 flex flex-col justify-between">
                    <div class="flex items-center justify-between mb-2 text-slate-400 text-[11px] border-b border-slate-800 pb-1">
                        <span class="text-cyan-400 font-semibold uppercase tracking-wider">v1 (Baseline) &bull; {html.escape(comp.v1_result.model)}</span>
                        <span>{comp.v1_result.total_tokens} tokens &bull; {comp.v1_result.latency_ms:.1f}ms</span>
                    </div>
                    <pre class="whitespace-pre-wrap leading-relaxed text-slate-200 overflow-x-auto flex-1">{v1_html}</pre>
                </div>

                <!-- Right: V2 -->
                <div class="bg-slate-950/80 rounded-lg p-4 border border-slate-800/80 flex flex-col justify-between">
                    <div class="flex items-center justify-between mb-2 text-slate-400 text-[11px] border-b border-slate-800 pb-1">
                        <span class="text-fuchsia-400 font-semibold uppercase tracking-wider">v2 (Candidate) &bull; {html.escape(comp.v2_result.model)}</span>
                        <span>{comp.v2_result.total_tokens} tokens &bull; {comp.v2_result.latency_ms:.1f}ms</span>
                    </div>
                    <pre class="whitespace-pre-wrap leading-relaxed text-slate-200 overflow-x-auto flex-1">{v2_html}</pre>
                </div>
            </div>

            <!-- Metric Scores Grid -->
            <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-2 pt-2">
                {''.join(score_badges)}
            </div>
        </div>
        """
        cards_html.append(card)

    failed_reasons_html = ""
    if not v.passed:
        items = "".join(f'<li class="text-xs text-rose-300 font-mono py-0.5">• {html.escape(f)}</li>' for f in v.failed_assertions)
        failed_reasons_html = f"""
        <div class="mb-6 p-4 rounded-xl bg-rose-950/40 border border-rose-800">
            <h4 class="text-sm font-semibold text-rose-200 mb-2 flex items-center gap-1.5">
                <span>⚠</span> Failed Assertions & Regressions:
            </h4>
            <ul class="list-none pl-1 space-y-1">{items}</ul>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>promptdiff Report &bull; {html.escape(report.v1_name)} vs {html.escape(report.v2_name)}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
        code, pre, .font-mono {{ font-family: 'JetBrains Mono', monospace; }}
    </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen p-6 md:p-10 antialiased">
    <div class="max-w-7xl mx-auto">
        <!-- Top Navbar / Header -->
        <header class="flex flex-col md:flex-row items-start md:items-center justify-between pb-6 mb-8 border-b border-slate-800 gap-4">
            <div>
                <div class="flex items-center gap-3 mb-1">
                    <span class="text-2xl font-bold tracking-tight bg-gradient-to-r from-cyan-400 via-indigo-400 to-fuchsia-400 bg-clip-text text-transparent">promptdiff</span>
                    <span class="text-xs uppercase tracking-wider font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">Regression Report</span>
                </div>
                <p class="text-xs text-slate-400 font-mono">
                    Comparing <span class="text-cyan-400 font-semibold">{html.escape(report.v1_name)}</span> &rarr; <span class="text-fuchsia-400 font-semibold">{html.escape(report.v2_name)}</span> &bull; {report.timestamp}
                </p>
            </div>
            
            <div class="flex items-center gap-3">
                <div class="px-4 py-2 rounded-lg border font-semibold text-xs flex items-center gap-2 {status_bg}">
                    <span class="text-sm font-bold">{status_icon}</span>
                    <span>{status_text}</span>
                </div>
            </div>
        </header>

        <!-- KPI Metrics Grid -->
        <section class="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
                <span class="text-xs font-medium text-slate-400 uppercase tracking-wider">Total Cost Delta</span>
                <div class="text-xl font-bold mt-1 {'text-rose-400' if v.cost_delta_pct > 0 else 'text-emerald-400'} font-mono">
                    {'+' if v.cost_delta_pct > 0 else ''}{v.cost_delta_pct:.1f}%
                </div>
                <span class="text-[11px] text-slate-500 font-mono">${v.total_cost_v1:.5f} &rarr; ${v.total_cost_v2:.5f}</span>
            </div>

            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
                <span class="text-xs font-medium text-slate-400 uppercase tracking-wider">Avg Latency Delta</span>
                <div class="text-xl font-bold mt-1 {'text-rose-400' if v.latency_delta_pct > 0 else 'text-emerald-400'} font-mono">
                    {'+' if v.latency_delta_pct > 0 else ''}{v.latency_delta_pct:.1f}%
                </div>
                <span class="text-[11px] text-slate-500 font-mono">{v.avg_latency_v1:.0f}ms &rarr; {v.avg_latency_v2:.0f}ms</span>
            </div>

            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
                <span class="text-xs font-medium text-slate-400 uppercase tracking-wider">Test Cases</span>
                <div class="text-xl font-bold mt-1 text-slate-100 font-mono">
                    {report.aggregate_stats.get('passed_cases', report.total_cases)} / {report.total_cases}
                </div>
                <span class="text-[11px] text-slate-500">Passed without regressions</span>
            </div>

            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
                <span class="text-xs font-medium text-slate-400 uppercase tracking-wider">Models</span>
                <div class="text-sm font-semibold mt-1.5 text-indigo-300 font-mono truncate">
                    {html.escape(report.model_v1)} &bull; {html.escape(report.model_v2)}
                </div>
                <span class="text-[11px] text-slate-500">Target inference runtime</span>
            </div>
        </section>

        {failed_reasons_html}

        <!-- Filter Controls -->
        <div class="flex items-center justify-between mb-4">
            <h2 class="text-sm font-semibold text-slate-300 uppercase tracking-wider">Test Case Diff Inspector</h2>
            <div class="flex gap-2">
                <button onclick="filterCases('all')" class="filter-btn active px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded text-xs font-medium transition">All ({report.total_cases})</button>
                <button onclick="filterCases('failed')" class="filter-btn px-3 py-1 bg-slate-800 hover:bg-slate-700 text-rose-300 rounded text-xs font-medium transition">Regressions Only</button>
                <button onclick="filterCases('passed')" class="filter-btn px-3 py-1 bg-slate-800 hover:bg-slate-700 text-emerald-300 rounded text-xs font-medium transition">Passed Only</button>
            </div>
        </div>

        <!-- Diff Cards -->
        <div id="diff-cards-container">
            {''.join(cards_html)}
        </div>

        <!-- Footer -->
        <footer class="mt-12 pt-6 border-t border-slate-900 text-center text-xs text-slate-600 font-mono">
            Generated with <a href="https://github.com/latryee/promptdiff" class="text-cyan-500 hover:underline">promptdiff</a> &bull; Production LLM Regression Testing Engine
        </footer>
    </div>

    <script>
        function filterCases(type) {{
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('bg-indigo-600', 'text-white'));
            event.target.classList.add('bg-indigo-600', 'text-white');
            
            const cards = document.querySelectorAll('.test-card');
            cards.forEach(card => {{
                if (type === 'all') {{
                    card.style.display = 'block';
                }} else if (type === 'failed') {{
                    card.style.display = card.classList.contains('case-failed') ? 'block' : 'none';
                }} else if (type === 'passed') {{
                    card.style.display = card.classList.contains('case-passed') ? 'block' : 'none';
                }}
            }});
        }}
    </script>
</body>
</html>
"""
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(html_content, encoding="utf-8")
    return str(dest.resolve())
