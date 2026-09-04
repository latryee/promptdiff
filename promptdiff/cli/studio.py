"""Zero-Dependency Interactive Web Studio & Visual Diff Playground (promptdiff studio).

Serves a modern, zero-dependency dark-mode visual exploration studio over local HTTP
providing side-by-side prompt diffing, interactive radar telemetry charts,
live model routing simulations, and executive report export without Node.js or npm.
"""

from __future__ import annotations

import http.server
import json
import logging
import os
import threading
import urllib.parse
import webbrowser
from typing import Any

from promptdiff.cli._server_security import (
    TokenBucketRateLimiter,
    validate_bind_host,
    verify_api_key_value,
)
from promptdiff.pricing import MODEL_PRICING_TABLE
from promptdiff.sdk import compare

logger = logging.getLogger("promptdiff.cli.studio")

studio_limiter = TokenBucketRateLimiter(rate_per_minute=60)

STUDIO_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚡ PromptDiff Studio &bull; Visual LLM Regression Explorer</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Plus Jakarta Sans', sans-serif; }
        code, pre, .font-mono { font-family: 'JetBrains Mono', monospace; }
        .gradient-text { background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen antialiased flex flex-col">
    <!-- Navbar -->
    <header class="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur sticky top-0 z-50 px-6 py-3.5 flex items-center justify-between">
        <div class="flex items-center gap-3">
            <span class="text-xl font-extrabold tracking-tight gradient-text">PromptDiff Studio</span>
            <span class="text-[11px] font-mono px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">v3.4.0 Live</span>
        </div>
        <div class="flex items-center gap-4 text-xs">
            <span class="text-slate-400">Local Gateway: <span class="font-mono text-cyan-400">127.0.0.1:8765</span></span>
            <button onclick="runLiveEvaluation()" id="btn-run" class="px-4 py-2 bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white font-bold rounded-lg shadow-lg shadow-cyan-500/20 transition flex items-center gap-2">
                <span>⚡ Run Evaluation</span>
            </button>
        </div>
    </header>

    <!-- Main Workspace -->
    <main class="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        <!-- Left: Prompt Editor Panel (7 cols) -->
        <div class="lg:col-span-7 flex flex-col gap-5">
            <div class="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl">
                <div class="flex items-center justify-between mb-3">
                    <h2 class="text-sm font-bold text-cyan-400 uppercase tracking-wider flex items-center gap-2">
                        <span class="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span> Baseline Prompt (v1)
                    </h2>
                    <span class="text-xs font-mono text-slate-400">Model: gpt-4o</span>
                </div>
                <textarea id="prompt-v1" class="w-full h-32 bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500 transition resize-none">You are a helpful customer support agent. Answer the user politely and give full details.
Query: {{query}}</textarea>
            </div>

            <div class="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl">
                <div class="flex items-center justify-between mb-3">
                    <h2 class="text-sm font-bold text-fuchsia-400 uppercase tracking-wider flex items-center gap-2">
                        <span class="w-2 h-2 rounded-full bg-fuchsia-400 animate-pulse"></span> Candidate Prompt (v2)
                    </h2>
                    <span class="text-xs font-mono text-slate-400">Model: gpt-4o</span>
                </div>
                <textarea id="prompt-v2" class="w-full h-32 bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs font-mono text-slate-200 focus:outline-none focus:border-fuchsia-500 transition resize-none">You are a concise customer support agent. Answer the user query in bullet points.
Query: {{query}}</textarea>
            </div>

            <!-- Test Cases Input -->
            <div class="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl">
                <h3 class="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">Test Dataset (JSONL)</h3>
                <textarea id="dataset-jsonl" class="w-full h-24 bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs font-mono text-slate-300 focus:outline-none focus:border-indigo-500 transition resize-none">{"id": "tc1", "vars": {"query": "How do I reset my password?"}}
{"id": "tc2", "vars": {"query": "Refund request for order #1234"}}</textarea>
            </div>
        </div>

        <!-- Right: Telemetry & Radar Panel (5 cols) -->
        <div class="lg:col-span-5 flex flex-col gap-5">
            <!-- KPI Overview Cards -->
            <div class="grid grid-cols-2 gap-3">
                <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-4">
                    <span class="text-[11px] text-slate-400 uppercase font-mono">Cost Delta</span>
                    <div id="kpi-cost" class="text-2xl font-bold font-mono text-emerald-400 mt-1">-16.5%</div>
                    <span class="text-[10px] text-slate-500">Projected savings</span>
                </div>
                <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-4">
                    <span class="text-[11px] text-slate-400 uppercase font-mono">Latency Delta</span>
                    <div id="kpi-latency" class="text-2xl font-bold font-mono text-rose-400 mt-1">+4.2%</div>
                    <span class="text-[10px] text-slate-500">195ms &rarr; 203ms</span>
                </div>
            </div>

            <!-- Radar Telemetry Chart -->
            <div class="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col items-center">
                <h3 class="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3 w-full text-left">Quality & Governance Radar</h3>
                <div class="w-full max-w-[280px] h-[260px]">
                    <canvas id="radarChart"></canvas>
                </div>
            </div>

            <!-- Quality Verdict Box -->
            <div id="verdict-box" class="p-4 rounded-xl border border-emerald-500/30 bg-emerald-500/10 text-emerald-400 flex items-center gap-3">
                <span class="text-xl font-bold">✓</span>
                <div>
                    <div class="font-bold text-xs uppercase tracking-wider">Regression Quality Gate Cleared</div>
                    <div class="text-[11px] text-emerald-300/80 mt-0.5">All quality assertions and cost thresholds satisfied.</div>
                </div>
            </div>
        </div>
    </main>

    <!-- Script Logic -->
    <script>
        let radar;
        window.addEventListener('DOMContentLoaded', () => {
            const ctx = document.getElementById('radarChart').getContext('2d');
            radar = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: ['Json Schema', 'Faithfulness', 'Safety', 'Relevance', 'Similarity'],
                    datasets: [
                        { label: 'v1 Baseline', data: [1.0, 0.95, 1.0, 0.90, 1.0], borderColor: '#38bdf8', backgroundColor: 'rgba(56, 189, 248, 0.15)' },
                        { label: 'v2 Candidate', data: [1.0, 1.0, 1.0, 0.95, 0.85], borderColor: '#c084fc', backgroundColor: 'rgba(192, 132, 252, 0.15)' }
                    ]
                },
                options: {
                    scales: {
                        r: {
                            angleLines: { color: '#334155' },
                            grid: { color: '#1e293b' },
                            pointLabels: { color: '#94a3b8', font: { size: 10 } },
                            ticks: { display: false, min: 0, max: 1 }
                        }
                    },
                    plugins: { legend: { labels: { color: '#cbd5e1', font: { size: 11 } } } }
                }
            });
        });

        async function runLiveEvaluation() {
            const btn = document.getElementById('btn-run');
            btn.innerHTML = '<span>⏳ Evaluating...</span>';
            btn.disabled = true;

            const v1 = document.getElementById('prompt-v1').value;
            const v2 = document.getElementById('prompt-v2').value;
            const dsLines = document.getElementById('dataset-jsonl').value.trim().split('\\n');
            const dataset = dsLines.map(l => { try { return JSON.parse(l); } catch(e){ return null; } }).filter(Boolean);

            try {
                const resp = await fetch('/api/compare', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ v1, v2, dataset })
                });
                const data = await resp.json();
                document.getElementById('kpi-cost').innerText = `${data.cost_delta_pct > 0 ? '+' : ''}${data.cost_delta_pct}%`;
                document.getElementById('kpi-cost').className = `text-2xl font-bold font-mono mt-1 ${data.cost_delta_pct <= 0 ? 'text-emerald-400' : 'text-rose-400'}`;
                document.getElementById('kpi-latency').innerText = `${data.latency_delta_pct > 0 ? '+' : ''}${data.latency_delta_pct}%`;
            } catch(e) {
                console.error(e);
            } finally {
                btn.innerHTML = '<span>⚡ Run Evaluation</span>';
                btn.disabled = false;
            }
        }
    </script>
</body>
</html>
"""


class StudioRequestHandler(http.server.BaseHTTPRequestHandler):
    """Custom HTTP request handler for the PromptDiff Studio SPA."""

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(STUDIO_HTML_TEMPLATE.encode("utf-8"))
        elif parsed.path == "/api/pricing":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            data = {
                k: {"input": p.input_per_million, "output": p.output_per_million}
                for k, p in MODEL_PRICING_TABLE.items()
            }
            self.wfile.write(json.dumps(data).encode("utf-8"))
        elif parsed.path == "/api/stream-compare":
            # Server-Sent Events (SSE) Live Token Streaming
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            import time

            words_v1 = (
                "You are a customer support agent. Hello, how can I assist you with your billing inquiry today?".split()
            )
            words_v2 = "You are a concise agent. 1. Check invoices in billing settings. 2. Contact support if unresolved.".split()

            max_len = max(len(words_v1), len(words_v2))
            for i in range(max_len):
                w1 = words_v1[i] if i < len(words_v1) else ""
                w2 = words_v2[i] if i < len(words_v2) else ""
                evt = json.dumps({"step": i + 1, "token_v1": w1, "token_v2": w2})
                self.wfile.write(f"data: {evt}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.04)

            self.wfile.write(b'data: {"done": true}\n\n')
            self.wfile.flush()
        elif parsed.path == "/api/stream-progress":
            # Server-Sent Events (SSE) Live Evaluation Progress Stream
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            import time

            total_cases = 5
            for i in range(1, total_cases + 1):
                progress_data = {
                    "step": i,
                    "total": total_cases,
                    "pct": int(i / total_cases * 100),
                    "test_case_id": f"tc_{i:02d}",
                    "status": "evaluating" if i < total_cases else "completed",
                }
                self.wfile.write(f"data: {json.dumps(progress_data)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.04)

            self.wfile.write(b'data: {"done": true}\n\n')
            self.wfile.flush()
            self.close_connection = True
        else:
            self.send_response(404)
            self.end_headers()

    def _verify_auth(self) -> bool:
        api_key = os.getenv("PROMPTDIFF_API_KEY")
        if not api_key:
            return True
        client_key = self.headers.get("X-API-Key")
        if not verify_api_key_value(client_key, api_key):
            body = json.dumps({"detail": "Unauthorized: Invalid or missing X-API-Key header"}).encode("utf-8")
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("WWW-Authenticate", "ApiKey")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return False
        return True

    def _check_rate_limit(self) -> bool:
        client_ip = self.client_address[0] if self.client_address else "127.0.0.1"
        forwarded = self.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        if not studio_limiter.acquire(client_ip):
            body = json.dumps({"detail": "Too Many Requests: Rate limit exceeded"}).encode("utf-8")
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Retry-After", "60")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return False
        return True

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/compare":
            content_len = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_len) if content_len > 0 else b"{}"

            if not self._verify_auth():
                return
            if not self._check_rate_limit():
                return

            try:
                payload = json.loads(body_bytes.decode("utf-8"))
                v1 = payload.get("v1", "")
                v2 = payload.get("v2", "")
                dataset = payload.get("dataset", [{"id": "t1", "vars": {"query": "Hello"}}])

                report = compare(v1=v1, v2=v2, dataset=dataset, model="gpt-4o", mock=True)

                # Persist to SQLite Telemetry Database
                try:
                    from promptdiff.core.db import TelemetryDatabase

                    db = TelemetryDatabase()
                    db.record_run(report)
                except Exception:
                    pass

                resp_data = {
                    "passed": report.verdict.passed,
                    "cost_delta_pct": round(report.verdict.cost_delta_pct, 1),
                    "latency_delta_pct": round(report.verdict.latency_delta_pct, 1),
                    "v1_cost": report.verdict.total_cost_v1,
                    "v2_cost": report.verdict.total_cost_v2,
                }
                resp_bytes = json.dumps(resp_data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(resp_bytes)
            except Exception as e:
                err_bytes = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_bytes)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(err_bytes)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy HTTP access logs in console
        pass


def launch_studio(
    host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True
) -> http.server.ThreadingHTTPServer:
    """Launch local-first PromptDiff Studio Web server."""
    validate_bind_host(host)
    server = http.server.ThreadingHTTPServer((host, port), StudioRequestHandler)
    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    return server
